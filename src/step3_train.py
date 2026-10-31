"""Step 3: one training run: drift fitted on the training rows, then fit and predict every eval set."""
import numpy as np

from classical import (
    fit_predict_classical,
    fit_predict_multilabel,
    fit_predict_naive,
    fit_predict_naive_multilabel,
    fit_predict_naive_reg,
    fit_predict_regressor,
)
from config import family_of
from drift import apply_drift
from retrieval import fit_predict_retrieval
from step2_torch import fit_predict_torch


def fit_predict(model, X, y, train_idx, test_idx, n_classes, cfg, seed):
    family = family_of(model)
    if cfg.get("kind") == "regression":
        if family == "naive":
            return fit_predict_naive_reg(y, train_idx, test_idx)
        if family == "classical":
            return fit_predict_regressor(model, X, y, train_idx, test_idx, seed)
        if family == "neural":
            return fit_predict_torch(model, X, y, train_idx, test_idx, n_classes, cfg, seed)
        return fit_predict_retrieval(model, X, y, train_idx, test_idx, n_classes, cfg, seed)
    if cfg.get("kind") == "multilabel":
        if family == "naive":
            return fit_predict_naive_multilabel(y, train_idx, test_idx, seed)
        if family == "classical":
            return fit_predict_multilabel(model, X, y, train_idx, test_idx, seed)
        if family == "neural":
            return fit_predict_torch(model, X, y, train_idx, test_idx, n_classes, cfg, seed)
        return fit_predict_retrieval(model, X, y, train_idx, test_idx, n_classes, cfg, seed)
    if family == "naive":
        return fit_predict_naive(X, y, train_idx, test_idx, n_classes, seed)
    if family == "classical":
        return fit_predict_classical(model, X, y, train_idx, test_idx, n_classes, seed)
    if family == "neural":
        return fit_predict_torch(model, X, y, train_idx, test_idx, n_classes, cfg, seed)
    return fit_predict_retrieval(model, X, y, train_idx, test_idx, n_classes, cfg, seed)


def train_run(model, drift, data, cfg, seed, train_idx, evals):
    """Return {eval split: (y_true, scores (n, n_classes))} for one fit."""
    y = data["y"]
    X = apply_drift(drift, data["X"], data["dates"], data["times"], train_idx)
    eval_idx = np.concatenate(list(evals.values()))
    scores = np.asarray(fit_predict(model, X, y, train_idx, eval_idx, len(data["classes"]), cfg, seed))
    out, start = {}, 0
    for name, idx in evals.items():
        out[name] = (y[idx], scores[start:start + len(idx)])
        start += len(idx)
    return out
