"""Step 5: aggregate the cell CSVs in results/runs/ into results/tables/*.csv.

all_runs.csv          every task, family, model, split, smooth, drift: mean, population std, n_runs.
{family}_{split}.csv  family: naive, classical, neural, ir; split: cv, holdout, lodo.
best_{split}.csv      grain "overall": best e2e (classical + neural) and best ir model per task,
                      searched over every model, smooth and drift;
                      grain "cell": the same inside each smooth x drift cell;
                      grain "model": each model's best smooth x drift.
ir_wins.csv           for each split, at the overall and cell grains, the metrics on which the best
                      ir model beats the best e2e model.
"Best" is the highest mean accuracy.
"""
import pandas as pd

from config import DRIFTS, EVAL_SPLITS, MODELS, RUNS_DIR, SMOOTHS, TABLES_DIR, TASKS

METRICS = ["acc", "f1", "auc"]
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


def load_summary():
    """Mean and population std (ddof=0, as in ICDM) over the runs of every cell."""
    files = [f for f in RUNS_DIR.glob("*.csv") if len(f.stem.split("__")) == 5]
    if not files:
        raise SystemExit(f"No cell files in {RUNS_DIR}")
    runs = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    g = runs.groupby(KEYS)[METRICS]
    summary = g.mean().join(g.std(ddof=0), rsuffix="_std").join(g.size().rename("n_runs")).reset_index()
    return ordered(summary, ["task", "split", "family", "model", "smooth", "drift"])


def best_rows(df, group_cols):
    return df.sort_values("acc", ascending=False, kind="stable").groupby(group_cols).head(1)


def best_table(summary, split):
    s = summary[summary["split"] == split]
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

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    for old in TABLES_DIR.glob("*.csv"):
        old.unlink()
    for name, df in tables.items():
        path = TABLES_DIR / f"{name}.csv"
        df.round(3).to_csv(path, index=False)
        print(f"[step5] {path.relative_to(TABLES_DIR.parent.parent)} ({len(df)} rows)")
    return tables
