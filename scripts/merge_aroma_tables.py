"""Merge finished aroma cells into results/tables without dropping the published rows."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from step4_analyze import collect_selected  # noqa: E402
from step5_visualize import merge_aroma_tables  # noqa: E402

TASKS = [
    "aroma_cluster_6",
    "aroma_families_12",
    "aroma_family_8",
    "aroma_descriptors_13",
]

if __name__ == "__main__":
    missing = collect_selected(TASKS)
    print(f"[merge] incomplete model cells: {missing}")
    merge_aroma_tables(TASKS)
