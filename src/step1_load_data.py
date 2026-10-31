"""Step 1: load raw traces and labels, z-score, optional LOWESS.

One row per wine (first measurement per `code`, as in ICDM).
Z-score is per (channel, time point) over the task rows, as in ICDM.
"""
import os

import numpy as np
import pandas as pd
from statsmodels.nonparametric.smoothers_lowess import lowess

from config import PROCESSED_DIR, RAW_DIR


def load_raw():
    meta = pd.read_csv(RAW_DIR / "merged.csv")
    X = np.load(RAW_DIR / "sensor_3d_array.npy").astype(np.float64)
    keep = ~meta["code"].duplicated(keep="first")
    return X[keep.values], meta.loc[keep].reset_index(drop=True)


def make_labels(meta, cfg):
    """Return (y int array, mask bool array, class names)."""
    if "target_columns" in cfg:
        Y = meta[cfg["target_columns"]].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy()
        mask = Y.sum(axis=1) > 0
        y = np.where(mask, Y.argmax(axis=1), -1)
        return y, mask, list(cfg["target_columns"])
    classes = cfg["classes"]
    col = meta[cfg["target"]]
    if isinstance(classes[0], (int, float)):
        col = pd.to_numeric(col, errors="coerce")
    mask = col.isin(classes).to_numpy()
    index = {c: i for i, c in enumerate(classes)}
    y = np.array([index[v] if m else -1 for v, m in zip(col, mask)], dtype=int)
    return y, mask, [str(c) for c in classes]


def zscore(X):
    return (X - X.mean(axis=0, keepdims=True)) / (X.std(axis=0, keepdims=True) + 1e-8)


def smooth_lowess(X, frac=0.05):
    out = np.empty_like(X)
    t = np.arange(X.shape[2], dtype=float)
    for i in range(X.shape[0]):
        for c in range(X.shape[1]):
            out[i, c] = lowess(X[i, c], t, frac=frac, return_sorted=False)
    return out


_loaded = {}


def load_task(cfg, smooth="lowess"):
    """Return dict X (N,C,T), y, classes, dates, times for one task. smooth: lowess | no_lowess.

    Traces are cached in data/processed/ and kept in memory for the rest of the process.
    """
    key = (cfg["name"], smooth)
    if key in _loaded:
        return _loaded[key]
    cache = PROCESSED_DIR / f"{cfg['name']}_{smooth}.npz"
    X_raw, meta = load_raw()
    y, mask, classes = make_labels(meta, cfg)
    meta = meta.loc[mask].reset_index(drop=True)
    y = y[mask]
    if cache.exists():
        X = np.load(cache)["X"]
    else:
        X = zscore(X_raw[mask])
        if smooth == "lowess":
            print(f"[step1] LOWESS on {X.shape} ...", flush=True)
            X = smooth_lowess(X, cfg["lowess_frac"])
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        tmp = cache.with_name(f"{cache.stem}.{os.getpid()}.tmp.npz")
        np.savez_compressed(tmp, X=X)
        os.replace(tmp, cache)
    print(f"[step1] {cfg['name']} smooth={smooth}: X={X.shape}, counts={np.bincount(y).tolist()}")
    _loaded[key] = {
        "X": X.astype(np.float32),
        "y": y,
        "classes": classes,
        "dates": meta["date"].astype(str).to_numpy(),
        "times": pd.to_datetime(meta["startTime"], utc=True, format="mixed").dt.tz_localize(None).to_numpy(),
    }
    return _loaded[key]


if __name__ == "__main__":
    from config import SMOOTHS, TASKS, load_config

    for task in TASKS:
        for smooth in SMOOTHS:
            load_task(load_config(task), smooth)
