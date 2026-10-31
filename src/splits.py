"""Splits used by the tables. Each run yields (run_id, seed, train_idx, evals).

cv / holdout: a row-level 4:1:1 shuffle (seed 42) sets aside one outer test set.
    StratifiedKFold(5) cuts train+val, and each fold is trained n_repeats times.
    Regression uses KFold with the same count. If the smaller class has fewer than
    5 rows in train+val, fewer folds are used and repeats increase so the cell
    still has n_folds * n_repeats fits.
    Every fit is scored on its left-out fold ("cv") and on the outer test ("holdout").
lodo: leave-one-day-out; train on the other days, score the left-out day.
"""
import math

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold


def outer_split(n, ratio=(4, 1, 1), seed=42):
    """Row-level shuffle; first n_test rows are the outer test set."""
    idx = np.arange(n)
    np.random.RandomState(seed).shuffle(idx)
    n_test = max(1, int(n * ratio[2] / sum(ratio)))
    return idx[n_test:], idx[:n_test]


def _fold_splits(y, train_val, cfg):
    """Return (list of (train, val) index pairs into train_val, repeats per fold)."""
    seed = cfg["random_state"]
    n_folds = cfg["n_folds"]
    if cfg.get("kind") == "regression":
        folds = list(KFold(n_splits=n_folds, shuffle=True, random_state=seed).split(train_val))
        return folds, cfg["n_repeats"]
    y_tv = np.asarray(y)[train_val]
    counts = np.bincount(y_tv.astype(int))
    counts = counts[counts > 0]
    n_use = min(n_folds, int(counts.min()))
    if n_use < 2:
        raise ValueError(f"Need at least 2 rows in each class to split, got {counts.tolist()}")
    repeats = cfg["n_repeats"] if n_use == n_folds else math.ceil(n_folds * cfg["n_repeats"] / n_use)
    folds = list(StratifiedKFold(n_splits=n_use, shuffle=True, random_state=seed).split(train_val, y_tv))
    return folds, repeats


def cv_runs(y, cfg):
    seed = cfg["random_state"]
    train_val, test = outer_split(len(y), cfg["train_val_test_ratio"], seed)
    folds, repeats = _fold_splits(y, train_val, cfg)
    total = cfg["n_folds"] * cfg["n_repeats"]
    produced = 0
    for fold, (tr, va) in enumerate(folds):
        for rep in range(repeats):
            if produced >= total:
                return
            produced += 1
            yield f"f{fold}r{rep}", seed + fold * 1000 + rep, train_val[tr], {"cv": train_val[va], "holdout": test}


def lodo_runs(dates, seed):
    for day in sorted(set(dates)):
        yield f"day{day}", seed, np.where(dates != day)[0], {"lodo": np.where(dates == day)[0]}


def _keep_labeled(idx, labeled):
    idx = np.asarray(idx)
    return idx[labeled[idx]]


def make_runs(split, data, cfg):
    """Build the runs. split_on tasks stratify on data['strata'], then drop unlabeled rows."""
    if split == "lodo":
        runs = list(lodo_runs(data["dates"], cfg["random_state"]))
    else:
        y_split = data["strata"] if "strata" in data else data["y"]
        runs = list(cv_runs(y_split, cfg))
    labeled = data.get("labeled")
    if labeled is None or np.asarray(labeled).all():
        return runs
    out = []
    for run_id, seed, train_idx, evals in runs:
        train_idx = _keep_labeled(train_idx, labeled)
        evals = {name: _keep_labeled(idx, labeled) for name, idx in evals.items()}
        out.append((run_id, seed, train_idx, evals))
    return out
