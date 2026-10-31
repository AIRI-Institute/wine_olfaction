"""Drift corrections applied to the traces of one run, fitted on its training rows only.

none:         traces unchanged.
snorm:        per-day z-score of each channel (mean and std over wines and time points) from
              that day's training rows; a day without training rows (LODO) uses pooled stats.
tod_residual: subtract a Ridge fit of the flattened trace on sin/cos of the UTC hour of day.
day_residual: subtract a Ridge fit of the flattened trace on a one-hot of the measurement
              day; a day unseen in training (LODO) gets the intercept only.
Residuals keep the trace scale (no L2 normalisation), so neural inputs stay z-score sized.
"""
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder


def snorm(X, dates, train_idx, eps=1e-6):
    tr_dates = dates[train_idx]
    pooled = (X[train_idx].mean(axis=(0, 2)), X[train_idx].std(axis=(0, 2)) + eps)
    stats = {}
    for d in np.unique(tr_dates):
        xd = X[train_idx[tr_dates == d]]
        stats[d] = (xd.mean(axis=(0, 2)), xd.std(axis=(0, 2)) + eps)
    out = np.empty_like(X)
    for d in np.unique(dates):
        rows = dates == d
        mu, sd = stats.get(d, pooled)
        out[rows] = (X[rows] - mu[:, None]) / sd[:, None]
    return out


def hour_features(times):
    t = np.asarray(times, dtype="datetime64[ns]")
    hours = (t - t.astype("datetime64[D]")).astype("timedelta64[s]").astype(np.float64) / 3600.0
    angle = 2.0 * np.pi * np.nan_to_num(hours) / 24.0
    return np.column_stack([np.sin(angle), np.cos(angle)])


def day_features(dates, train_idx):
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    enc.fit(dates[train_idx, None])
    return enc.transform(dates[:, None])


def residual(X, features, train_idx, alpha=1.0):
    Z = X.reshape(len(X), -1)
    ridge = Ridge(alpha=alpha).fit(features[train_idx], Z[train_idx])
    return (Z - ridge.predict(features)).reshape(X.shape)


def apply_drift(method, X, dates, times, train_idx):
    if method == "none":
        return X
    X64 = np.asarray(X, dtype=np.float64)
    if method == "snorm":
        out = snorm(X64, dates, train_idx)
    elif method == "tod_residual":
        out = residual(X64, hour_features(times), train_idx)
    elif method == "day_residual":
        out = residual(X64, day_features(dates, train_idx), train_idx)
    else:
        raise ValueError(f"Unknown drift {method}")
    return out.astype(X.dtype)
