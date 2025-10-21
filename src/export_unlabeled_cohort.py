"""Export unlabeled cohort from cluster assignments for pseudo-labeling.

Reads outputs/tables/cluster_assignments.csv (produced by src.clustering) and
writes a CSV of unlabeled, non-noise samples suitable for filtering the weak
pool during training.

Default behavior:
- Select rows where is_labeled == False and cluster_dbscan != -1
- Output columns: path, cluster_dbscan (and optionally cluster_kmeans)

Usage:
    python -m src.export_unlabeled_cohort \
      --assignments outputs/tables/cluster_assignments.csv \
      --method dbscan \
      --output outputs/tables/unlabeled_cohort_dbscan.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export unlabeled DBSCAN/KMeans cohort")
    parser.add_argument(
        "--assignments",
        type=Path,
        default=Path("outputs/tables/cluster_assignments.csv"),
        help="Path to cluster assignments CSV",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="dbscan",
        choices=["dbscan", "kmeans"],
        help="Which cluster labels to use for cohort selection",
    )
    parser.add_argument(
        "--cluster-id",
        type=int,
        default=None,
        help="Optional specific cluster ID to export (if omitted: all non-noise for DBSCAN)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/tables/unlabeled_cohort.csv"),
        help="Output CSV path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.assignments)
    if "is_labeled" not in df.columns or "path" not in df.columns:
        raise SystemExit("Assignments CSV must contain 'path' and 'is_labeled' columns")

    mask = ~df["is_labeled"].astype(bool)
    if args.method == "dbscan":
        if "cluster_dbscan" not in df.columns:
            raise SystemExit("Assignments CSV missing 'cluster_dbscan' column")
        if args.cluster_id is not None:
            mask &= df["cluster_dbscan"] == int(args.cluster_id)
        else:
            mask &= df["cluster_dbscan"] != -1
        cohort = df.loc[mask, ["path", "cluster_dbscan", "cluster_kmeans"]].copy()
    else:
        if "cluster_kmeans" not in df.columns:
            raise SystemExit("Assignments CSV missing 'cluster_kmeans' column")
        if args.cluster_id is not None:
            mask &= df["cluster_kmeans"] == int(args.cluster_id)
        cohort = df.loc[mask, ["path", "cluster_kmeans", "cluster_dbscan"]].copy()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    cohort.to_csv(args.output, index=False)
    print(f"Wrote cohort CSV with {len(cohort)} rows to {args.output}")


if __name__ == "__main__":
    main()
