"""Step 4: metrics per run.

One run writes results/runs/parts/<task>__<model>__<smooth>__<drift>/<split>__<run>.csv
(a cv run holds a "cv" row and a "holdout" row). When every run of a cell is present,
collect_cells writes results/runs/<task>__<model>__<eval split>__<smooth>__<drift>.csv.

Classification: accuracy, F1, roc_auc, and pr_auc for binary tasks.
F1 is binary for 2-class tasks and macro for more classes (ICDM).
roc_auc and pr_auc are empty for svm (hard labels only) and when the eval set has one class.
Multilabel accuracy is Hamming accuracy. subset_accuracy is the exact match of the label set.
precision_macro and precision_micro are filled for classification and multilabel.
Regression: r2 and mae. Classification columns are empty.
"""
import os

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    hamming_loss,
    mean_absolute_error,
    precision_score,
    r2_score,
    roc_auc_score,
)

from config import DRIFTS, MODELS, RUNS_DIR, SMOOTHS, TASKS, TRAIN_SPLITS, family_of, load_config
from splits import make_runs
from step1_load_data import load_task

PARTS_DIR = RUNS_DIR / "parts"


def _macro_average_precision(y_true, scores, present):
    aps = []
    for c in present:
        yt = (np.asarray(y_true) == c).astype(int)
        if yt.min() == yt.max():
            continue
        aps.append(average_precision_score(yt, scores[:, c]))
    return float(np.mean(aps)) if aps else np.nan


def run_metrics(y_true, scores, n_classes, kind="classification", pr_auc_macro=False):
    empty = {
        "acc": np.nan, "f1": np.nan, "roc_auc": np.nan, "pr_auc": np.nan,
        "subset_accuracy": np.nan, "precision_macro": np.nan, "precision_micro": np.nan,
        "r2": np.nan, "mae": np.nan,
    }
    if kind == "regression":
        pred = np.asarray(scores, dtype=float).reshape(-1)
        empty["r2"] = r2_score(y_true, pred)
        empty["mae"] = mean_absolute_error(y_true, pred)
        return empty
    out = dict(empty)
    if kind == "multilabel":
        y_true = np.asarray(y_true).astype(int)
        pred = (np.asarray(scores) >= 0.5).astype(int)
        out["acc"] = 1.0 - hamming_loss(y_true, pred)
        out["subset_accuracy"] = accuracy_score(y_true, pred)
        out["f1"] = f1_score(y_true, pred, average="macro", zero_division=0)
        out["precision_macro"] = precision_score(y_true, pred, average="macro", zero_division=0)
        out["precision_micro"] = precision_score(y_true, pred, average="micro", zero_division=0)
        present = [c for c in range(y_true.shape[1]) if len(np.unique(y_true[:, c])) == 2]
        try:
            if present:
                cols = np.asarray(present)
                out["roc_auc"] = roc_auc_score(y_true[:, cols], scores[:, cols], average="macro")
                out["pr_auc"] = average_precision_score(y_true[:, cols], scores[:, cols], average="macro")
        except ValueError:
            pass
        return out
    pred = scores.argmax(axis=1)
    out["acc"] = accuracy_score(y_true, pred)
    average = "binary" if n_classes == 2 else "macro"
    out["f1"] = f1_score(y_true, pred, average=average, zero_division=0)
    out["precision_macro"] = precision_score(y_true, pred, average=average, zero_division=0)
    out["precision_micro"] = precision_score(y_true, pred, average="micro", zero_division=0)
    present = np.unique(y_true)
    try:
        if n_classes == 2 and len(present) == 2:
            out["roc_auc"] = roc_auc_score(y_true, scores[:, 1])
            out["pr_auc"] = average_precision_score(y_true, scores[:, 1])
        elif len(present) > 2:
            s = scores[:, present] + 1e-12
            out["roc_auc"] = roc_auc_score(y_true, s / s.sum(axis=1, keepdims=True), multi_class="ovr", labels=present)
            if pr_auc_macro:
                out["pr_auc"] = _macro_average_precision(y_true, scores, present)
    except ValueError:
        pass
    return out


def part_path(task, model, smooth, drift, split, run_id):
    return PARTS_DIR / f"{task}__{model}__{smooth}__{drift}" / f"{split}__{run_id}.csv"


def cell_path(task, model, split, smooth, drift):
    return RUNS_DIR / f"{task}__{model}__{split}__{smooth}__{drift}.csv"


def save_part(task, model, smooth, drift, split, run_id, results, n_classes, kind="classification",
              pr_auc_macro=False):
    rows = []
    for eval_split, (y_true, scores) in results.items():
        m = run_metrics(y_true, scores, n_classes, kind, pr_auc_macro=pr_auc_macro)
        if kind != "regression" and model == "svm":
            m["roc_auc"] = np.nan
            m["pr_auc"] = np.nan
        rows.append({"task": task, "family": family_of(model), "model": model, "split": eval_split,
                     "smooth": smooth, "drift": drift, "run": run_id, "n_test": len(y_true), **m})
    path = part_path(task, model, smooth, drift, split, run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.stem}.{os.getpid()}.tmp")
    pd.DataFrame(rows).to_csv(tmp, index=False)
    os.replace(tmp, path)
    return rows


def collect_selected(tasks, smooth="no_lowess", drift="none"):
    """Write cell CSVs for one smooth and drift. Return the number of incomplete model cells."""
    n_missing = 0
    for task in tasks:
        cfg = load_config(task)
        data = load_task(cfg, smooth)
        for split in TRAIN_SPLITS:
            run_ids = [r[0] for r in make_runs(split, data, cfg)]
            for model in MODELS:
                paths = [part_path(task, model, smooth, drift, split, r) for r in run_ids]
                if not all(p.exists() for p in paths):
                    n_missing += 1
                    continue
                df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
                for eval_split, group in df.groupby("split"):
                    group.to_csv(cell_path(task, model, eval_split, smooth, drift), index=False)
    return n_missing


def collect_cells(repeats=None):
    """Write every complete cell CSV; return the number of incomplete cells."""
    n_missing = 0
    for task in TASKS:
        cfg = load_config(task)
        if repeats is not None:
            cfg["n_repeats"] = repeats
        for smooth in SMOOTHS:
            data = load_task(cfg, smooth)
            for split in TRAIN_SPLITS:
                run_ids = [r[0] for r in make_runs(split, data, cfg)]
                for model in MODELS:
                    for drift in DRIFTS:
                        paths = [part_path(task, model, smooth, drift, split, r) for r in run_ids]
                        if not all(p.exists() for p in paths):
                            n_missing += 1
                            continue
                        df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
                        for eval_split, g in df.groupby("split"):
                            g.to_csv(cell_path(task, model, eval_split, smooth, drift), index=False)
    return n_missing
