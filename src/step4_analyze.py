"""Step 4: accuracy, F1, ROC-AUC per run.

One run writes results/runs/parts/<task>__<model>__<smooth>__<drift>/<split>__<run>.csv
(a cv run holds a "cv" row and a "holdout" row). When every run of a cell is present,
collect_cells writes results/runs/<task>__<model>__<eval split>__<smooth>__<drift>.csv.

F1 is binary for 2-class tasks and macro for more classes (ICDM).
ROC-AUC is empty for svm (hard labels only) and when the eval set has one class.
"""
import os

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from config import DRIFTS, MODELS, RUNS_DIR, SMOOTHS, TASKS, TRAIN_SPLITS, family_of, load_config
from splits import make_runs
from step1_load_data import load_task

PARTS_DIR = RUNS_DIR / "parts"


def run_metrics(y_true, scores, n_classes):
    pred = scores.argmax(axis=1)
    out = {
        "acc": accuracy_score(y_true, pred),
        "f1": f1_score(y_true, pred, average="binary" if n_classes == 2 else "macro", zero_division=0),
        "auc": np.nan,
    }
    present = np.unique(y_true)
    try:
        if n_classes == 2 and len(present) == 2:
            out["auc"] = roc_auc_score(y_true, scores[:, 1])
        elif len(present) > 2:
            s = scores[:, present] + 1e-12
            out["auc"] = roc_auc_score(y_true, s / s.sum(axis=1, keepdims=True), multi_class="ovr", labels=present)
    except ValueError:
        pass
    return out


def part_path(task, model, smooth, drift, split, run_id):
    return PARTS_DIR / f"{task}__{model}__{smooth}__{drift}" / f"{split}__{run_id}.csv"


def cell_path(task, model, split, smooth, drift):
    return RUNS_DIR / f"{task}__{model}__{split}__{smooth}__{drift}.csv"


def save_part(task, model, smooth, drift, split, run_id, results, n_classes):
    rows = []
    for eval_split, (y_true, scores) in results.items():
        m = run_metrics(y_true, scores, n_classes)
        if model == "svm":
            m["auc"] = np.nan
        rows.append({"task": task, "family": family_of(model), "model": model, "split": eval_split,
                     "smooth": smooth, "drift": drift, "run": run_id, "n_test": len(y_true), **m})
    path = part_path(task, model, smooth, drift, split, run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.stem}.{os.getpid()}.tmp")
    pd.DataFrame(rows).to_csv(tmp, index=False)
    os.replace(tmp, path)
    return rows


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
