"""Naive and classical models on flattened traces (C*T features)."""
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import LinearSVC, LinearSVR
from xgboost import XGBClassifier, XGBRegressor

from config import N_JOBS


def _positive_proba(clf, Z):
    """Probability of class 1. A label missing from training scores 0."""
    proba = clf.predict_proba(Z)
    classes = list(clf.classes_)
    if 1 not in classes:
        return np.zeros(len(Z))
    return proba[:, classes.index(1)]


def fit_predict_naive_multilabel(y, train_idx, test_idx, seed):
    """Per-label stratified dummy. Returns (n_test, n_labels) scores."""
    scores = np.zeros((len(test_idx), y.shape[1]))
    ztr = np.zeros((len(train_idx), 1))
    zte = np.zeros((len(test_idx), 1))
    for c in range(y.shape[1]):
        yc = y[train_idx, c]
        if yc.min() == yc.max():
            scores[:, c] = yc[0]
            continue
        clf = DummyClassifier(strategy="stratified", random_state=seed)
        clf.fit(ztr, yc)
        scores[:, c] = _positive_proba(clf, zte)
    return scores


def fit_predict_naive(X, y, train_idx, test_idx, n_classes, seed):
    clf = DummyClassifier(strategy="stratified", random_state=seed)
    clf.fit(np.zeros((len(train_idx), 1)), y[train_idx])
    return _full_proba(clf.predict_proba(np.zeros((len(test_idx), 1))), clf.classes_, n_classes)


def fit_predict_naive_reg(y, train_idx, test_idx):
    reg = DummyRegressor(strategy="mean")
    reg.fit(np.zeros((len(train_idx), 1)), y[train_idx])
    return reg.predict(np.zeros((len(test_idx), 1)))


def fit_predict_regressor(name, X, y, train_idx, test_idx, seed):
    """Return (n_test,) predictions. svm is LinearSVR, the regression counterpart of LinearSVC."""
    Z = X.reshape(len(X), -1)
    ztr, zte, ytr = Z[train_idx], Z[test_idx], y[train_idx]
    if name == "plsda":
        pls = PLSRegression(n_components=max(1, min(8, len(train_idx) - 1)))
        pls.fit(ztr, ytr)
        return np.asarray(pls.predict(zte)).reshape(-1)
    if name == "rf":
        reg = RandomForestRegressor(n_estimators=200, random_state=seed, n_jobs=N_JOBS)
    elif name == "xgb":
        reg = XGBRegressor(n_estimators=200, max_depth=4, n_jobs=N_JOBS, verbosity=0, random_state=seed)
    elif name == "svm":
        reg = LinearSVR(random_state=seed, max_iter=4000)
    else:
        raise ValueError(name)
    reg.fit(ztr, ytr)
    return reg.predict(zte)


def fit_predict_classical(name, X, y, train_idx, test_idx, n_classes, seed):
    """Return (n_test, n_classes) scores. svm returns one-hot votes (no ROC-AUC)."""
    Z = X.reshape(len(X), -1)
    ztr, zte, ytr = Z[train_idx], Z[test_idx], y[train_idx]
    if name == "plsda":
        pls = PLSRegression(n_components=max(1, min(8, n_classes, len(train_idx) - 1)))
        pls.fit(ztr, np.eye(n_classes)[ytr])
        return np.asarray(pls.predict(zte))
    if name == "rf":
        clf = RandomForestClassifier(n_estimators=200, random_state=seed, class_weight="balanced", n_jobs=N_JOBS)
    elif name == "xgb":
        # xgb needs labels 0..k-1 present in training
        present, ytr = np.unique(ytr, return_inverse=True)
        clf = XGBClassifier(n_estimators=200, max_depth=4, n_jobs=N_JOBS, verbosity=0, random_state=seed)
        clf.fit(ztr, ytr)
        return _full_proba(clf.predict_proba(zte), present, n_classes)
    elif name == "svm":
        clf = LinearSVC(class_weight="balanced", random_state=seed, max_iter=4000)
        clf.fit(ztr, ytr)
        return np.eye(n_classes)[clf.predict(zte)]
    else:
        raise ValueError(name)
    clf.fit(ztr, ytr)
    return _full_proba(clf.predict_proba(zte), clf.classes_, n_classes)


def fit_predict_multilabel(name, X, y, train_idx, test_idx, seed):
    """One score per label. svm returns hard 0/1 votes. A constant training label is copied."""
    Z = X.reshape(len(X), -1)
    ztr, zte, ytr = Z[train_idx], Z[test_idx], y[train_idx]
    n_labels = y.shape[1]
    if name == "plsda":
        pls = PLSRegression(n_components=max(1, min(8, n_labels, len(train_idx) - 1)))
        pls.fit(ztr, ytr)
        return np.asarray(pls.predict(zte))
    scores = np.zeros((len(test_idx), n_labels))
    for c in range(n_labels):
        yc = ytr[:, c]
        if yc.min() == yc.max():
            scores[:, c] = yc[0]
            continue
        if name == "rf":
            clf = RandomForestClassifier(n_estimators=200, random_state=seed, class_weight="balanced", n_jobs=N_JOBS)
            clf.fit(ztr, yc)
            scores[:, c] = _positive_proba(clf, zte)
        elif name == "xgb":
            clf = XGBClassifier(n_estimators=200, max_depth=4, n_jobs=N_JOBS, verbosity=0, random_state=seed)
            clf.fit(ztr, yc)
            scores[:, c] = _positive_proba(clf, zte)
        elif name == "svm":
            clf = LinearSVC(class_weight="balanced", random_state=seed, max_iter=4000)
            clf.fit(ztr, yc)
            scores[:, c] = clf.predict(zte)
        else:
            raise ValueError(name)
    return scores


def _full_proba(p, seen_classes, n_classes):
    """Place columns of classes seen in training into a full (n, n_classes) matrix."""
    out = np.zeros((p.shape[0], n_classes))
    out[:, np.asarray(seen_classes, dtype=int)] = p
    return out
