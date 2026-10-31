"""Retrieval: embed traces, find cosine neighbours in the training gallery, vote.

knn_pca:         PCA(32) on flattened traces, weighted vote of k = n_classes neighbours.
cosine_ssl:      time-crop SimCLR encoder trained on the gallery, nearest neighbour (k=1).
cosine_curvefit: exponential rise/recovery parameters per channel, nearest neighbour (k=1).
All transforms are fit on gallery (train) rows only.
"""
import numpy as np
from scipy.optimize import curve_fit
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

K_MAX = 13
_curvefit_cache = {}


def l2_normalize(z):
    return z / np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-8)


def impute_train_median(z, train_idx):
    z = z.copy()
    med = np.nanmedian(z[train_idx], axis=0)
    med = np.where(np.isnan(med), 0.0, med)
    rows, cols = np.where(np.isnan(z))
    z[rows, cols] = med[cols]
    return z


def pca_embed(X, train_idx, dim=32):
    z = X.reshape(len(X), -1).astype(np.float64)
    scaler = StandardScaler().fit(z[train_idx])
    z = scaler.transform(z)
    pca = PCA(n_components=min(dim, len(train_idx) - 1, z.shape[1]), random_state=0).fit(z[train_idx])
    return pca.transform(z)


def _exp_sat(t, a, tau, b):
    return b + a * (1.0 - np.exp(-t / np.clip(tau, 1e-3, 1e3)))


def _exp_rec(t, a, tau, b):
    return b + a * np.exp(-t / np.clip(tau, 1e-3, 1e3))


def curvefit_features(X):
    """Rise (first half) and recovery (second half) exponential fits: 6 params per channel."""
    key = (X.shape, float(X.sum()))
    if key in _curvefit_cache:
        return _curvefit_cache[key]
    n, c, t = X.shape
    mid = t // 2
    t_sat, t_rec = np.arange(mid, dtype=float), np.arange(t - mid, dtype=float)
    out = np.full((n, c * 6), np.nan)
    for i in range(n):
        for ch in range(c):
            ys, yr = X[i, ch, :mid], X[i, ch, mid:]
            try:
                out[i, ch * 6:ch * 6 + 3] = curve_fit(
                    _exp_sat, t_sat, ys, p0=(ys[-1] - ys[0], 20.0, ys[0]), maxfev=400)[0]
            except Exception:
                pass
            try:
                out[i, ch * 6 + 3:ch * 6 + 6] = curve_fit(
                    _exp_rec, t_rec, yr, p0=(yr[0] - yr[-1], 30.0, yr[-1]), maxfev=400)[0]
            except Exception:
                pass
    _curvefit_cache[key] = out
    return out


def curvefit_embed(X, train_idx):
    z = impute_train_median(curvefit_features(X), train_idx)
    return StandardScaler().fit(z[train_idx]).transform(z)


def ssl_embed(X, train_idx, seed=0, crop=64, dim=64, epochs=40, batch=64, lr=1e-3, temp=0.2):
    """SimCLR with two random time crops of the same trace as the positive pair."""
    import torch
    from torch import nn
    from torch.nn import functional as F

    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n, c, t = X.shape
    mu = X[train_idx].mean(axis=(0, 2), keepdims=True)
    sd = X[train_idx].std(axis=(0, 2), keepdims=True) + 1e-6
    Xn = ((X - mu) / sd).astype(np.float32)

    net = nn.Sequential(
        nn.Conv1d(c, 32, 5, padding=2), nn.ReLU(),
        nn.Conv1d(32, 64, 5, padding=2), nn.ReLU(),
        nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(64, dim),
    ).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    def crops(idx):
        starts = rng.randint(0, max(1, t - crop + 1), size=len(idx))
        return torch.from_numpy(np.stack([Xn[i, :, s:s + crop] for i, s in zip(idx, starts)])).to(dev)

    ids = np.array(train_idx)
    for _ in range(epochs):
        rng.shuffle(ids)
        for s in range(0, len(ids), batch):
            b = ids[s:s + batch]
            if len(b) < 4:
                continue
            z = F.normalize(net(torch.cat([crops(b), crops(b)])), dim=1)
            sim = (z @ z.T / temp).masked_fill(torch.eye(2 * len(b), dtype=torch.bool, device=dev), -1e9)
            target = torch.cat([torch.arange(len(b)) + len(b), torch.arange(len(b))]).to(dev)
            loss = F.cross_entropy(sim, target)
            opt.zero_grad()
            loss.backward()
            opt.step()
    start = max(0, (t - crop) // 2)
    with torch.no_grad():
        return net(torch.from_numpy(Xn[:, :, start:start + crop]).to(dev)).cpu().numpy()


def embed(name, X, train_idx, cfg, seed):
    if name == "knn_pca":
        z = pca_embed(X, train_idx, cfg["pca_dim"])
    elif name == "cosine_ssl":
        z = ssl_embed(X, train_idx, seed=0)
    elif name == "cosine_curvefit":
        z = curvefit_embed(X, train_idx)
    else:
        raise ValueError(name)
    return l2_normalize(z)


def vote(z, y, train_idx, test_idx, n_classes, k):
    """Cosine top-k neighbours in the gallery; class scores = summed positive similarity."""
    sims = z[test_idx] @ z[train_idx].T
    k = min(k, len(train_idx))
    top = np.argsort(-sims, axis=1)[:, :k]
    scores = np.zeros((len(test_idx), n_classes))
    for q in range(len(test_idx)):
        for j in top[q]:
            scores[q, y[train_idx[j]]] += max(sims[q, j], 0.0)
    scores[scores.sum(axis=1) == 0, 0] = 1.0
    return scores / scores.sum(axis=1, keepdims=True)


def fit_predict_retrieval(name, X, y, train_idx, test_idx, n_classes, cfg, seed):
    z = embed(name, X, train_idx, cfg, seed)
    k = min(n_classes, K_MAX) if name == "knn_pca" else 1
    return vote(z, y, train_idx, test_idx, n_classes, k)
