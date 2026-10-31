"""Step 1: load raw traces and labels, z-score, optional LOWESS.

One row per wine (first measurement per `code`, as in ICDM).
Z-score is per (channel, time point) over the task rows, as in ICDM.
"""
import os

import h5py
import numpy as np
import pandas as pd
from statsmodels.nonparametric.smoothers_lowess import lowess

from config import PROCESSED_DIR, RAW_DIR

CHANNELS = [
    "ZnO-1", "ZnO-2", "ZnO-3",
    "ZnO/In-1", "ZnO/In-2", "ZnO/In-3",
    "ZnO/Ni-1", "ZnO/Ni-2", "ZnO/Ni-3",
    "ZnO/Ce-1", "ZnO/Ce-2",
    "ZnO/Ag-1", "ZnO/Ag-2",
]

# Highest wine_quality that is still a positive (low) label. k=4 flags nothing:
# every score is >= 61.33, so the flag is 1 only strictly below that value.
QUALITY_CUTOFF = {0: 82.11, 1: 76.73, 2: 70.33, 3: 65.63, 4: 61.33}


def wine_quality_bin(q, k):
    """1 when wine_quality is on the low side of the cutoff for this k.

    k=0..3 use the highest score that counts as low.
    k=4 uses a strict cutoff below every score in the file, so the result is all zeros.
    """
    q = np.asarray(q, dtype=float)
    k = int(k)
    if k == 4:
        return (q < QUALITY_CUTOFF[4]).astype(int)
    return (q <= QUALITY_CUTOFF[k]).astype(int)


def check_quality_bins(meta):
    """Fail if wine_quality_bin(k=0) does not reproduce wine_quality_low_mean."""
    q = pd.to_numeric(meta["wine_quality"], errors="coerce")
    got = pd.to_numeric(meta["wine_quality_low_mean"], errors="coerce").to_numpy()
    pred = wine_quality_bin(q, 0)
    if np.isnan(got).any() or not np.array_equal(pred, got.astype(int)):
        raise SystemExit("wine_quality_bin(k=0) disagrees with wine_quality_low_mean")


def _hdf5_strings(values):
    return [v.decode() if isinstance(v, bytes) else str(v) for v in values]


def load_raw():
    meta = pd.read_csv(RAW_DIR / "metadata.csv")
    check_quality_bins(meta)
    with h5py.File(RAW_DIR / "sensor_traces.h5", "r") as f:
        X = np.asarray(f["traces"][:], dtype=np.float64)
        row = np.asarray(f["row"][:])
        code = np.asarray(f["code"][:], dtype=np.float64)
        channel = _hdf5_strings(f["channel"][:])
    expected_code = pd.to_numeric(meta["code"], errors="coerce").to_numpy(dtype=np.float64)
    if (
        X.shape != (len(meta), len(CHANNELS), 189)
        or not np.array_equal(row, np.arange(len(meta)))
        or not np.array_equal(code, expected_code)
        or channel != CHANNELS
    ):
        raise SystemExit("sensor_traces.h5 row, code, or channel does not match metadata.csv")
    keep = ~meta["code"].duplicated(keep="first")
    return X[keep.values], meta.loc[keep].reset_index(drop=True)


def _parse_list(raw):
    text = "" if pd.isna(raw) else str(raw).strip()
    if text in ("", "[]"):
        return []
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1].strip()
        if not text:
            return []
    return [part.strip() for part in text.split(",") if part.strip()]


def make_labels(meta, cfg):
    """Return (y array, mask bool array, class names).

    Regression y is the average of strength_low and strength_high on this row.
    quality_k builds the low-quality flag from wine_quality.
    Multilabel y is a 0/1 matrix. An empty list is the all-negative row.
    """
    if cfg.get("kind") == "regression":
        low = pd.to_numeric(meta["strength_low"], errors="coerce")
        high = pd.to_numeric(meta["strength_high"], errors="coerce")
        y = ((low + high) / 2).to_numpy(dtype=float)
        return y, np.isfinite(y), []
    if cfg.get("kind") == "multilabel":
        classes = [str(c) for c in cfg["classes"]]
        index = {c: i for i, c in enumerate(classes)}
        Y = np.zeros((len(meta), len(classes)), dtype=int)
        for i, raw in enumerate(meta[cfg["target"]].tolist()):
            for name in _parse_list(raw):
                if name not in index:
                    raise SystemExit(f"unknown {cfg['target']} label {name!r}")
                Y[i, index[name]] = 1
        return Y, np.ones(len(meta), dtype=bool), classes
    if "quality_k" in cfg:
        q = pd.to_numeric(meta["wine_quality"], errors="coerce").to_numpy(dtype=float)
        y = np.where(np.isfinite(q), wine_quality_bin(q, cfg["quality_k"]), -1)
        return y.astype(int), np.isfinite(q), [str(c) for c in cfg["classes"]]
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


def _strata(meta, cfg):
    """Class index used to build the shared fold. split_on names a task yaml."""
    from config import load_config

    split_cfg = cfg if cfg.get("name") == cfg.get("split_on") else load_config(cfg["split_on"])
    strata, ok, _ = make_labels(meta, split_cfg)
    if not ok.all() or np.ndim(strata) != 1:
        raise SystemExit(f"split_on {cfg['split_on']} is not one class on every row")
    return strata.astype(int)


def load_task(cfg, smooth="lowess"):
    """Return dict X (N,C,T), y, classes, dates, times for one task. smooth: lowess | no_lowess.

    Traces are cached in data/processed/ and kept in memory for the rest of the process.
    A task with split_on keeps every deduplicated row, so its indices match that split.
    Rows that fail the task label stay in place and are marked unlabeled.
    """
    key = (cfg["name"], smooth)
    if key in _loaded:
        return _loaded[key]
    cache = PROCESSED_DIR / f"{cfg['name']}_{smooth}.npz"
    X_raw, meta = load_raw()
    y, labeled, classes = make_labels(meta, cfg)
    strata = _strata(meta, cfg) if "split_on" in cfg else None
    mask = np.ones(len(meta), dtype=bool) if strata is not None else labeled
    meta = meta.loc[mask].reset_index(drop=True)
    y = y[mask]
    labeled = np.asarray(labeled)[mask]
    if strata is not None:
        strata = strata[mask]
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
    if X.shape[0] != len(meta):
        raise SystemExit(f"{cache.name} has {X.shape[0]} rows, metadata has {len(meta)}")
    if cfg.get("kind") == "regression":
        print(f"[step1] {cfg['name']} smooth={smooth}: X={X.shape}, y mean={y.mean():.4f} n={len(y)}")
    elif cfg.get("kind") == "multilabel":
        print(f"[step1] {cfg['name']} smooth={smooth}: X={X.shape}, positives={y.sum(axis=0).tolist()}")
    else:
        counted = y[labeled] if labeled is not None else y
        print(f"[step1] {cfg['name']} smooth={smooth}: X={X.shape}, counts={np.bincount(counted.astype(int)).tolist()}")
    pack = {
        "X": X.astype(np.float32),
        "y": y,
        "classes": classes,
        "dates": meta["date"].astype(str).to_numpy(),
        "times": pd.to_datetime(meta["startTime"], utc=True, format="mixed").dt.tz_localize(None).to_numpy(),
        "labeled": labeled,
    }
    if strata is not None:
        pack["strata"] = strata
    _loaded[key] = pack
    return _loaded[key]


if __name__ == "__main__":
    from config import SMOOTHS, TASKS, load_config

    for task in TASKS:
        cfg = load_config(task)
        for smooth in cfg.get("smooths", SMOOTHS):
            load_task(cfg, smooth)
