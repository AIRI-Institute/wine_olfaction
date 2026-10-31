"""Build the complete cell CSVs from results/runs/parts/, then write results/tables/*.csv."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from step4_analyze import collect_cells  # noqa: E402
from step5_visualize import write_tables  # noqa: E402

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--repeats", type=int, help="n_repeats used by run_all.py (smoke tests)")
    a = p.parse_args()
    print(f"[write_tables] incomplete cells: {collect_cells(a.repeats)}")
    write_tables()
