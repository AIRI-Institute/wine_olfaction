"""Paths, grid factors and config loading."""
import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
CONFIG_DIR = ROOT / "configs"
RUNS_DIR = ROOT / "results" / "runs"
TABLES_DIR = ROOT / "results" / "tables"

TASKS = ["color2", "color3", "abv_b", "aroma_markup_ce", "low_q"]
NAIVE = ["Naive"]
CLASSICAL = ["rf", "xgb", "plsda", "svm"]
NEURAL = ["MLSTMfcn", "Transformer", "CNNLSTM4", "CNN1DClassifier", "LSTMNet", "EmissionCNN1DLSTM"]
RETRIEVAL = ["knn_pca", "cosine_ssl", "cosine_curvefit"]
MODELS = NAIVE + CLASSICAL + NEURAL + RETRIEVAL
SMOOTHS = ["lowess", "no_lowess"]
DRIFTS = ["none", "snorm", "tod_residual", "day_residual"]
TRAIN_SPLITS = ["cv", "lodo"]           # a cv fit is also scored on the outer test ("holdout")
EVAL_SPLITS = ["cv", "holdout", "lodo"]

# Slurm limits a job to its allocated cores; os.cpu_count() reports the whole node.
N_JOBS = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))


def family_of(model: str) -> str:
    if model in NAIVE:
        return "naive"
    if model in CLASSICAL:
        return "classical"
    if model in NEURAL:
        return "neural"
    if model in RETRIEVAL:
        return "ir"
    raise ValueError(f"Unknown model {model}")


def load_config(task: str) -> dict:
    """base.yaml merged with configs/<task>.yaml."""
    cfg = yaml.safe_load((CONFIG_DIR / "base.yaml").read_text())
    cfg.update(yaml.safe_load((CONFIG_DIR / f"{task}.yaml").read_text()))
    return cfg
