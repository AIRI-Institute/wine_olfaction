"""Splits used by the tables. Each run yields (run_id, seed, train_idx, evals).

cv / holdout: a row-level 4:1:1 shuffle (seed 42) sets aside one outer test set.
    StratifiedKFold(5) cuts train+val, and each fold is trained n_repeats times.
    Every fit is scored on its left-out fold ("cv") and on the outer test ("holdout").
lodo: leave-one-day-out; train on the other days, score the left-out day.
"""
import numpy as np
from sklearn.model_selection import StratifiedKFold


def outer_split(n, ratio=(4, 1, 1), seed=42):
    """Row-level shuffle; first n_test rows are the outer test set."""
    idx = np.arange(n)
    np.random.RandomState(seed).shuffle(idx)
    n_test = max(1, int(n * ratio[2] / sum(ratio)))
    return idx[n_test:], idx[:n_test]


def cv_runs(y, cfg):
    seed = cfg["random_state"]
    train_val, test = outer_split(len(y), cfg["train_val_test_ratio"], seed)
    skf = StratifiedKFold(n_splits=cfg["n_folds"], shuffle=True, random_state=seed)
    for fold, (tr, va) in enumerate(skf.split(train_val, y[train_val])):
        for rep in range(cfg["n_repeats"]):
            yield f"f{fold}r{rep}", seed + fold * 1000 + rep, train_val[tr], {"cv": train_val[va], "holdout": test}


def lodo_runs(dates, seed):
    for day in sorted(set(dates)):
        yield f"day{day}", seed, np.where(dates != day)[0], {"lodo": np.where(dates == day)[0]}


def make_runs(split, data, cfg):
    if split == "lodo":
        return list(lodo_runs(data["dates"], cfg["random_state"]))
    return list(cv_runs(data["y"], cfg))
