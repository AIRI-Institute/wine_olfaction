"""PyTorch train / predict helpers (ICDM: one-hot targets, BCE-with-logits, last epoch)."""
import numpy as np
import torch
import torch.nn as nn

from config import N_JOBS
from models import create_model

torch.set_num_threads(N_JOBS)


def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def fit_predict_torch(model_name, X, y, train_idx, test_idx, n_classes, cfg, seed):
    """Train on train_idx, return class probabilities for test_idx."""
    set_seed(seed)
    dev = device()
    Xt = torch.tensor(X, dtype=torch.float32)
    Y = torch.tensor(np.eye(n_classes, dtype=np.float32)[y])
    model = create_model(model_name, in_channels=X.shape[1], num_classes=n_classes).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    loss_fn = nn.BCEWithLogitsLoss()
    bs = cfg["batch_size"]
    tr = np.asarray(train_idx)
    for _ in range(cfg["n_epochs"]):
        model.train()
        order = tr[torch.randperm(len(tr)).numpy()]
        for s in range(0, len(order), bs):
            b = order[s:s + bs]
            xb, yb = Xt[b].to(dev), Y[b].to(dev)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
    model.eval()
    probs = []
    with torch.no_grad():
        te = np.asarray(test_idx)
        for s in range(0, len(te), bs):
            probs.append(torch.softmax(model(Xt[te[s:s + bs]].to(dev)), dim=1).cpu().numpy())
    return np.concatenate(probs)
