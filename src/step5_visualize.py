"""Step 5: aggregate the cell CSVs in results/runs/ into results/tables/*.csv.

all_runs.csv          every task, family, model, split, smooth, drift: mean, population std, n_runs.
                      Metrics: acc, f1, roc_auc, pr_auc, subset_accuracy, precision_macro,
                      precision_micro, r2, mae. Empty when the task does not use them.
{family}_{split}.csv  family: naive, classical, neural, ir; split: cv, holdout, lodo.
best_{split}.csv      grain "overall": best e2e (classical + neural) and best ir model per task,
                      searched over every model, smooth and drift;
                      grain "cell": the same inside each smooth x drift cell;
                      grain "model": each model's best smooth x drift.
                      Regression rows have no accuracy and are left out.
ir_wins.csv           for each split, at the overall and cell grains, the metrics on which the best
                      ir model beats the best e2e model.
low_q_threshold.csv   "Low-quality detection vs. quality threshold" on the holdout split:
                      k, score_cutoff, pos_rate, best neural model by PR-AUC, roc_auc, pr_auc.
"Best" in best_*.csv is the highest mean accuracy.
"""
import os

import numpy as np
import pandas as pd

from config import DRIFTS, EVAL_SPLITS, MODELS, RUNS_DIR, SMOOTHS, TABLES_DIR, TASKS
from step1_load_data import QUALITY_CUTOFF, load_raw, wine_quality_bin

METRICS = [
    "acc", "f1", "roc_auc", "pr_auc", "subset_accuracy", "precision_macro", "precision_micro", "r2", "mae",
]
FAMILIES = ["naive", "classical", "neural", "ir"]
SIDE = {"classical": "e2e", "neural": "e2e", "ir": "ir"}
KEYS = ["task", "family", "model", "split", "smooth", "drift"]
STATS = [c for m in METRICS for c in (m, f"{m}_std")]
ORDER = {name: {v: i for i, v in enumerate(values)} for name, values in
         [("task", TASKS), ("model", MODELS), ("smooth", SMOOTHS), ("drift", DRIFTS),
          ("split", EVAL_SPLITS), ("family", FAMILIES), ("grain", ["overall", "cell", "model"]),
          ("side", ["e2e", "ir"])]}


def ordered(df, cols):
    return df.sort_values(cols, key=lambda s: s.map(ORDER[s.name]) if s.name in ORDER else s)


def _read_runs(path):
    df = pd.read_csv(path)
    if "auc" in df.columns and "roc_auc" not in df.columns:
        df = df.rename(columns={"auc": "roc_auc"})
    for metric in METRICS:
        if metric not in df.columns:
            df[metric] = np.nan
    return df


def load_summary():
    """Mean and population std (ddof=0, as in ICDM) over the runs of every cell."""
    files = [f for f in RUNS_DIR.glob("*.csv") if len(f.stem.split("__")) == 5]
    if not files:
        raise SystemExit(f"No cell files in {RUNS_DIR}")
    runs = pd.concat([_read_runs(f) for f in files], ignore_index=True)
    g = runs.groupby(KEYS)[METRICS]
    summary = g.mean().join(g.std(ddof=0), rsuffix="_std").join(g.size().rename("n_runs")).reset_index()
    return ordered(summary, ["task", "split", "family", "model", "smooth", "drift"])


def best_rows(df, group_cols):
    return df.sort_values("acc", ascending=False, kind="stable").groupby(group_cols).head(1)


def best_table(summary, split):
    s = summary[summary["split"] == split]
    s = s[np.isfinite(s["acc"])]
    s = s.assign(side=s["family"].map(SIDE)).dropna(subset=["side"])
    out = pd.concat([
        best_rows(s, ["task", "side"]).assign(grain="overall"),
        best_rows(s, ["task", "smooth", "drift", "side"]).assign(grain="cell"),
        best_rows(s, ["task", "model"]).assign(grain="model"),
    ])
    out = ordered(out, ["grain", "task", "smooth", "drift", "side", "model"])
    return out[["grain", "side", "task", "family", "model", "smooth", "drift"] + STATS + ["n_runs"]]


def wins(best, split):
    """Rows where the best ir model has a higher metric than the best e2e model."""
    cols = ["split", "grain", "task", "smooth", "drift", "metric", "e2e_model", "e2e_smooth", "e2e_drift",
            "ir_model", "ir_smooth", "ir_drift", "e2e", "ir", "advantage", "acc_e2e", "acc_ir"]
    rows = []
    for grain, keys in (("overall", ["task"]), ("cell", ["task", "smooth", "drift"])):
        b = best[best["grain"] == grain]
        e2e = b[b["side"] == "e2e"].set_index(keys, drop=False)
        ir = b[b["side"] == "ir"].set_index(keys, drop=False)
        for key in e2e.index.intersection(ir.index):
            e, i = e2e.loc[key], ir.loc[key]
            for m in METRICS:
                if pd.notna(e[m]) and pd.notna(i[m]) and i[m] > e[m]:
                    row = {"split": split, "grain": grain, "metric": m,
                           "e2e_model": e["model"], "e2e_smooth": e["smooth"], "e2e_drift": e["drift"],
                           "ir_model": i["model"], "ir_smooth": i["smooth"], "ir_drift": i["drift"],
                           "e2e": e[m], "ir": i[m], "advantage": i[m] - e[m],
                           "acc_e2e": e["acc"], "acc_ir": i["acc"]}
                    row.update(dict(zip(keys, key if isinstance(key, tuple) else (key,))))
                    rows.append(row)
    return pd.DataFrame(rows, columns=cols)


def threshold_table(summary):
    """Low-quality detection vs. quality threshold, holdout split.

    best_model is the neural model with the highest mean PR-AUC across smoothing and drift.
    k=4 has no positive wines and no trained model.
    pos_rate is the percent of kept wines labeled positive.
    """
    _, meta = load_raw()
    q = meta["wine_quality"]
    tasks = {0: "low_q", 1: "low_q_k1", 2: "low_q_k2", 3: "low_q_k3"}
    rows = []
    for k in range(5):
        y = wine_quality_bin(q, k)
        row = {"k": k, "score_cutoff": QUALITY_CUTOFF[k], "pos_rate": 100 * float(np.mean(y)),
               "best_model": "", "roc_auc": np.nan, "pr_auc": np.nan}
        task = tasks.get(k)
        if task is not None:
            s = summary[(summary["task"] == task) & (summary["split"] == "holdout") & (summary["family"] == "neural")]
            s = s[s["pr_auc"].notna()]
            if len(s):
                best = s.sort_values(["pr_auc", "roc_auc"], ascending=False, kind="stable").iloc[0]
                row["best_model"] = best["model"]
                row["roc_auc"] = best["roc_auc"]
                row["pr_auc"] = best["pr_auc"]
        rows.append(row)
    return pd.DataFrame(rows)


def write_tables():
    summary = load_summary()
    tables = {"all_runs": summary}
    for split in EVAL_SPLITS:
        for family in FAMILIES:
            s = summary[(summary["split"] == split) & (summary["family"] == family)]
            tables[f"{family}_{split}"] = s[["task", "model", "smooth", "drift"] + STATS + ["n_runs"]]
    for split in EVAL_SPLITS:
        tables[f"best_{split}"] = best_table(summary, split)
    tables["ir_wins"] = pd.concat([wins(tables[f"best_{s}"], s) for s in EVAL_SPLITS], ignore_index=True)
    tables["low_q_threshold"] = threshold_table(summary)

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    for old in TABLES_DIR.glob("*.csv"):
        old.unlink()
    for name, df in tables.items():
        path = TABLES_DIR / f"{name}.csv"
        df.round(3).to_csv(path, index=False)
        print(f"[step5] {path.relative_to(TABLES_DIR.parent.parent)} ({len(df)} rows)")
    return tables


def _row_key(df):
    return df[KEYS].astype(str).agg("|".join, axis=1)


def aggregate_cells(paths):
    """Mean and population std over the runs stored in these cell CSVs."""
    runs = pd.concat([_read_runs(path) for path in paths], ignore_index=True)
    grouped = runs.groupby(KEYS, sort=False)[METRICS]
    return grouped.mean().join(grouped.std(ddof=0), rsuffix="_std").join(grouped.size().rename("n_runs")).reset_index()


def merge_aroma_tables(tasks, smooth="no_lowess", drift="none"):
    """Add finished aroma cells to the published tables.

    Rows already in all_runs.csv stay. A new cell replaces the row with the same key.
    low_q_threshold.csv is left untouched. This does not rebuild tables from an empty runs directory.
    """
    paths, expected = [], 0
    for task in tasks:
        for model in MODELS:
            for split in EVAL_SPLITS:
                expected += 1
                path = RUNS_DIR / f"{task}__{model}__{split}__{smooth}__{drift}.csv"
                if path.exists():
                    paths.append(path)
    if not paths:
        raise SystemExit(f"no cell files in {RUNS_DIR}")
    print(f"[merge] {len(paths)}/{expected} cell files")
    new = aggregate_cells(paths)
    old = pd.read_csv(TABLES_DIR / "all_runs.csv")
    for metric in METRICS:
        if metric not in old.columns:
            old[metric] = np.nan
        if f"{metric}_std" not in old.columns:
            old[f"{metric}_std"] = np.nan
    old = old.loc[~_row_key(old).isin(set(_row_key(new)))].copy()
    summary = ordered(pd.concat([old, new], ignore_index=True),
                      ["task", "split", "family", "model", "smooth", "drift"])
    summary = summary[KEYS + STATS + ["n_runs"]]
    tables = {"all_runs": summary}
    for split in EVAL_SPLITS:
        for family in FAMILIES:
            s = summary[(summary["split"] == split) & (summary["family"] == family)]
            tables[f"{family}_{split}"] = s[["task", "model", "smooth", "drift"] + STATS + ["n_runs"]]
        tables[f"best_{split}"] = best_table(summary, split)
    tables["ir_wins"] = pd.concat([wins(tables[f"best_{s}"], s) for s in EVAL_SPLITS], ignore_index=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        path = TABLES_DIR / f"{name}.csv"
        tmp = path.with_name(f"{path.stem}.{os.getpid()}.tmp.csv")
        df.round(3).to_csv(tmp, index=False)
        os.replace(tmp, path)
        print(f"[merge] {path.name} ({len(df)} rows)")
    return tables
