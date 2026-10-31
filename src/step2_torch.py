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
    """Train on train_idx. Classification returns class probabilities; regression returns a scalar."""
    set_seed(seed)
    dev = device()
    Xt = torch.tensor(X, dtype=torch.float32)
    regression = cfg.get("kind") == "regression"
    multilabel = cfg.get("kind") == "multilabel"
    if regression:
        Y = torch.tensor(np.asarray(y, dtype=np.float32)).unsqueeze(1)
        n_out = 1
        loss_fn = nn.MSELoss()
    elif multilabel:
        Y = torch.tensor(np.asarray(y, dtype=np.float32))
        n_out = Y.shape[1]
        loss_fn = nn.BCEWithLogitsLoss()
    else:
        Y = torch.tensor(np.eye(n_classes, dtype=np.float32)[y])
        n_out = n_classes
        loss_fn = nn.BCEWithLogitsLoss()
    model = create_model(model_name, in_channels=X.shape[1], num_classes=n_out).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
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
    out = []
    with torch.no_grad():
        te = np.asarray(test_idx)
        for s in range(0, len(te), bs):
            pred = model(Xt[te[s:s + bs]].to(dev))
            if multilabel:
                pred = torch.sigmoid(pred)
            elif not regression:
                pred = torch.softmax(pred, dim=1)
            out.append(pred.cpu().numpy())
    pred = np.concatenate(out)
    return pred.reshape(-1) if regression else pred
