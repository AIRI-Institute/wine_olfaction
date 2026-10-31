"""Naive and classical models on flattened traces (C*T features)."""
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

from config import N_JOBS


def fit_predict_naive(X, y, train_idx, test_idx, n_classes, seed):
    clf = DummyClassifier(strategy="stratified", random_state=seed)
    clf.fit(np.zeros((len(train_idx), 1)), y[train_idx])
    return _full_proba(clf.predict_proba(np.zeros((len(test_idx), 1))), clf.classes_, n_classes)


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


def _full_proba(p, seen_classes, n_classes):
    """Place columns of classes seen in training into a full (n, n_classes) matrix."""
    out = np.zeros((p.shape[0], n_classes))
    out[:, np.asarray(seen_classes, dtype=int)] = p
    return out
