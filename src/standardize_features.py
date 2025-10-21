from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def standardize_embeddings(
    embeddings_path: Path,
    csv_path: Path,
    output_path: Path,
) -> None:
    if not embeddings_path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")
    if not csv_path.exists():
        raise FileNotFoundError(f"Embeddings CSV not found: {csv_path}")

    logging.info("Loading embeddings from %s", embeddings_path)
    E = np.load(embeddings_path)
    if E.ndim != 2:
        raise ValueError(f"Embeddings must be 2D [N, D], got shape {E.shape}")

    logging.info("Loading metadata from %s", csv_path)
    df = pd.read_csv(csv_path)
    required_cols = {"index", "path", "bucket", "label"}
    missing = required_cols - set(df.columns)
    if missing:
        raise KeyError(f"Embeddings CSV missing columns: {', '.join(sorted(missing))}")

    # Ensure rows align with embedding rows by sorting on the explicit index column
    df = df.sort_values("index").reset_index(drop=True)
    if len(df) != E.shape[0]:
        raise ValueError(
            f"Row count mismatch between CSV ({len(df)}) and embeddings ({E.shape[0]})"
        )

    logging.info("Fitting StandardScaler and transforming features")
    scaler = StandardScaler()
    Z = scaler.fit_transform(E.astype(np.float32))

    paths = df["path"].astype(str).to_numpy()
    is_labeled = (df["bucket"].astype(str) == "labeled").to_numpy()
    labels = df["label"].fillna("").astype(str)
    labels = labels.where(is_labeled, "").to_numpy()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        features=Z.astype(np.float32),
        paths=paths,
        is_labeled=is_labeled,
        labels=labels,
        scaler_mean=np.asarray(scaler.mean_, dtype=np.float32),
        scaler_scale=np.asarray(scaler.scale_, dtype=np.float32),
    )

    logging.info(
        "Wrote standardized bundle: %s (N=%d, D=%d)", output_path, Z.shape[0], Z.shape[1]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Standardize embeddings and build feature bundle for clustering. "
            "Consumes outputs/features/embeddings.{npy,csv} and writes "
            "outputs/features/standardized_features.npz by default."
        )
    )
    parser.add_argument(
        "--embeddings-npy",
        type=Path,
        default=Path("outputs/features/embeddings.npy"),
        help="Path to embeddings .npy file",
    )
    parser.add_argument(
        "--embeddings-csv",
        type=Path,
        default=Path("outputs/features/embeddings.csv"),
        help="Path to embeddings CSV file (paths + labels)",
    )
    parser.add_argument(
        "--output-npz",
        type=Path,
        default=Path("outputs/features/standardized_features.npz"),
        help="Path to write the standardized feature bundle",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    standardize_embeddings(args.embeddings_npy, args.embeddings_csv, args.output_npz)


if __name__ == "__main__":
    main()
