"""Train every missing run of the grid, then write results/tables/*.csv.

Grid: 5 tasks x 14 models x smooth {lowess, no_lowess} x drift {none, snorm, tod_residual, day_residual}
      x split {cv: 25 fits scored on the left-out fold and the outer test, lodo: 6 left-out days}.
One run writes one file in results/runs/parts/; finished runs are skipped.
Several copies may run at once (Slurm arrays): each run is claimed with an exclusive lock file
in results/runs/claims/, and a claim older than --claim-ttl minutes counts as abandoned.

Examples:
    python scripts/run_all.py
    python scripts/run_all.py --tasks color2 --models rf knn_pca --drifts none --epochs 1 --repeats 1
"""
import argparse
import os
import socket
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from config import DRIFTS, MODELS, RUNS_DIR, SMOOTHS, TASKS, TRAIN_SPLITS, load_config  # noqa: E402
from splits import make_runs  # noqa: E402
from step1_load_data import load_task  # noqa: E402
from step3_train import train_run  # noqa: E402
from step4_analyze import PARTS_DIR, collect_cells, part_path, save_part  # noqa: E402
from step5_visualize import write_tables  # noqa: E402

CLAIMS_DIR = RUNS_DIR / "claims"


def _allowed(requested, cfg, key):
    """Yaml smooths/drifts narrow the command-line grid. A task without the key keeps the request."""
    limited = cfg.get(key)
    if not limited:
        return list(requested)
    return [value for value in requested if value in limited]


def units(a):
    """Every (cfg, data, task, model, smooth, drift, split, run) of the selected grid."""
    out = []
    for task in a.tasks:
        cfg = load_config(task)
        if a.epochs is not None:
            cfg["n_epochs"] = a.epochs
        if a.repeats is not None:
            cfg["n_repeats"] = a.repeats
        for smooth in _allowed(a.smooths, cfg, "smooths"):
            data = load_task(cfg, smooth)
            for split in a.splits:
                runs = make_runs(split, data, cfg)
                for drift in _allowed(a.drifts, cfg, "drifts"):
                    for model in a.models:
                        out += [(cfg, data, task, model, smooth, drift, split, run) for run in runs]
    return out


def claim(path, ttl_min):
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        try:
            stale = time.time() - path.stat().st_mtime > ttl_min * 60
        except FileNotFoundError:
            return False
        if stale:
            path.unlink(missing_ok=True)
            return claim(path, ttl_min)
        return False
    os.write(fd, f"{socket.gethostname()} {os.environ.get('SLURM_JOB_ID', os.getpid())}\n".encode())
    os.close(fd)
    return True


def main(a):
    todo = units(a)
    n_workers = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))
    worker = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)) - int(os.environ.get("SLURM_ARRAY_TASK_MIN", 0))
    start = worker * len(todo) // n_workers
    todo = todo[start:] + todo[:start]
    print(f"[run_all] {len(todo)} runs in the grid, worker {worker + 1}/{n_workers}", flush=True)
    n_done = 0
    for cfg, data, task, model, smooth, drift, split, (run_id, seed, train_idx, evals) in todo:
        part = part_path(task, model, smooth, drift, split, run_id)
        if part.exists():
            continue
        lock = CLAIMS_DIR / part.relative_to(PARTS_DIR).with_suffix(".claim")
        if not claim(lock, a.claim_ttl):
            continue
        t0 = time.time()
        try:
            if not part.exists():
                results = train_run(model, drift, data, cfg, seed, train_idx, evals)
                rows = save_part(task, model, smooth, drift, split, run_id, results, len(data["classes"]),
                                 cfg.get("kind", "classification"), pr_auc_macro=cfg.get("pr_auc") == "macro")
                n_done += 1
                bits = []
                for r in rows:
                    if r["r2"] == r["r2"]:
                        bits.append(f"{r['split']}:r2={r['r2']:.3f}")
                    else:
                        bits.append(f"{r['split']}:acc={r['acc']:.3f}")
                print(f"[run] {task} {model} {smooth} {drift} {split} {run_id} {time.time() - t0:.0f}s "
                      + " ".join(bits), flush=True)
        except Exception:
            print(f"[fail] {task} {model} {smooth} {drift} {split} {run_id}", flush=True)
            traceback.print_exc()
        finally:
            lock.unlink(missing_ok=True)
    print(f"[run_all] trained {n_done} runs", flush=True)
    if not a.no_tables:
        n_missing = collect_cells(a.repeats)
        print(f"[run_all] incomplete cells: {n_missing}")
        write_tables()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", nargs="+", default=TASKS)
    p.add_argument("--models", nargs="+", default=MODELS)
    p.add_argument("--smooths", nargs="+", default=SMOOTHS, choices=SMOOTHS)
    p.add_argument("--drifts", nargs="+", default=DRIFTS, choices=DRIFTS)
    p.add_argument("--splits", nargs="+", default=TRAIN_SPLITS, choices=TRAIN_SPLITS)
    p.add_argument("--epochs", type=int, help="override n_epochs (smoke tests)")
    p.add_argument("--repeats", type=int, help="override n_repeats (smoke tests)")
    p.add_argument("--claim-ttl", type=float, default=120, help="minutes before a claim counts as abandoned")
    p.add_argument("--no-tables", action="store_true", help="skip steps 4-5 (parallel cluster workers)")
    main(p.parse_args())
