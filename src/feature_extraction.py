"""Feature extraction pipeline for MRI brain cancer dataset.

This module performs deterministic preprocessing and extracts
fixed-length embeddings from a pretrained ResNet backbone. The
implementation follows the workflow outlined for Task 2 of the
semi-supervised image processing project:

1. Discover all images in the labeled and unlabeled buckets.
2. Apply resize / center-crop / normalization transforms compatible
   with ImageNet pretraining.
3. Run the frozen backbone in inference mode to produce embeddings for
   each image, logging any decode failures along the way.
4. Persist the embeddings, metadata, and summary artifacts to the
   ``outputs`` directory tree.

Why features first?
- Separating feature extraction from training lets students reuse the same
    representation for multiple downstream tasks (clustering, classifiers),
    speeding iteration and making experiments comparable.
- We use a pretrained ResNet-18 as a “universal” feature extractor: it has
    learned general edge/texture/shape detectors from ImageNet that transfer
    surprisingly well to medical images after simple normalisation.

Run directly as a script to regenerate embeddings::

    python -m src.feature_extraction --data-dir mri_dataset_brain_cancer_oc

The script can be configured with additional CLI flags; run with
``--help`` for details.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Iterable

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from torch import nn
from torchvision import models, transforms

# ---------------------------------------------------------------------------
# Constants and configuration
# ---------------------------------------------------------------------------

DEFAULT_DATA_DIR = Path("mri_dataset_brain_cancer_oc")
DEFAULT_OUTPUT_ROOT = Path("outputs")
FEATURE_OUTPUT_DIR = DEFAULT_OUTPUT_ROOT / "features"
LOG_OUTPUT_DIR = DEFAULT_OUTPUT_ROOT / "logs"
NOTE_OUTPUT_DIR = DEFAULT_OUTPUT_ROOT / "notes"
LOG_PATH = LOG_OUTPUT_DIR / "feature_extraction.log"
EMBEDDING_ARRAY_PATH = FEATURE_OUTPUT_DIR / "embeddings.npy"
EMBEDDING_CSV_PATH = FEATURE_OUTPUT_DIR / "embeddings.csv"
METADATA_PATH = FEATURE_OUTPUT_DIR / "metadata.json"
SUMMARY_NOTE_PATH = NOTE_OUTPUT_DIR / "feature_summary.md"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
TARGET_RESIZE = 256
TARGET_CROP = 224
BATCH_SIZE = 32
NEIGHBOR_SAMPLE = 8
RNG_SEED = 42

LABELED_BUCKET = "avec_labels"
UNLABELED_BUCKET = "sans_label"

BACKBONE_NAME = "torchvision.resnet18"
BACKBONE_WEIGHTS = "ResNet18_Weights.IMAGENET1K_V1"
BACKBONE_LAYER = "global_avg_pool"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ImageRecord:
    """Representation of a single image in the dataset."""

    absolute_path: Path
    relative_path: Path
    bucket: str
    label: Optional[str]


@dataclass
class ExtractionResults:
    """Container for outputs generated during extraction."""

    embeddings: np.ndarray
    records: List[ImageRecord]
    failures: List[Path]
    per_file_times: List[float]


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def configure_logging(verbose: bool = False) -> None:
    """Configure logging to both stdout and the log file."""

    LOG_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    handlers: List[logging.Handler] = [
        logging.FileHandler(LOG_PATH, mode="w", encoding="utf-8"),
        logging.StreamHandler(),
    ]
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=handlers,
    )


def discover_image_records(data_dir: Path) -> List[ImageRecord]:
    """Enumerate labeled and unlabeled images as :class:`ImageRecord`s."""

    # We store both absolute and relative paths and whether a
    # sample is labeled. This makes it easy to build tables and trace results
    # back to files later (e.g., nearest-neighbor spot checks).

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    records: List[ImageRecord] = []
    # Labeled bucket: expect label folders inside ``avec_labels``
    labeled_root = data_dir / LABELED_BUCKET
    if labeled_root.exists():
        for label_dir in sorted(p for p in labeled_root.iterdir() if p.is_dir()):
            label = label_dir.name
            for image_path in sorted(label_dir.rglob("*")):
                if image_path.is_file():
                    relative = image_path.relative_to(data_dir)
                    records.append(
                        ImageRecord(
                            absolute_path=image_path,
                            relative_path=relative,
                            bucket="labeled",
                            label=label,
                        )
                    )
    else:
        logging.warning("Labeled bucket missing at %s", labeled_root)

    # Unlabeled bucket: files directly under ``sans_label``
    unlabeled_root = data_dir / UNLABELED_BUCKET
    if unlabeled_root.exists():
        for image_path in sorted(unlabeled_root.rglob("*")):
            if image_path.is_file():
                relative = image_path.relative_to(data_dir)
                records.append(
                    ImageRecord(
                        absolute_path=image_path,
                        relative_path=relative,
                        bucket="unlabeled",
                        label=None,
                    )
                )
    else:
        logging.warning("Unlabeled bucket missing at %s", unlabeled_root)

    if not records:
        raise RuntimeError(f"No image files discovered under {data_dir}")

    logging.info(
        "Discovered %d images (labeled=%d, unlabeled=%d)",
        len(records),
        sum(1 for r in records if r.bucket == "labeled"),
        sum(1 for r in records if r.bucket == "unlabeled"),
    )
    return records


def build_transform() -> transforms.Compose:
    """Return preprocessing transform with deterministic resizing."""

    # These transforms deliberately mirror ImageNet training
    # Why 224 with a 256→224 resize-crop?
    # - ResNet-18 was pretrained on ImageNet at 224×224; keeping the
    #   same scale preserves learned inductive biases and improves transfer.
    # - Resize to 256 then center-crop to 224 is a standard ImageNet
    #   preprocessing pattern that reduces border artifacts and standardises
    #   content framing.
    # - This resolution balances detail and throughput/VRAM. If your task
    #   requires finer structures, consider larger crops consistently across
    #   data and model.
    # stats so the backbone’s early layers receive inputs they expect. We
    # avoid randomness here to keep embeddings reproducible.

    return transforms.Compose(
        [
            transforms.Resize(TARGET_RESIZE),
            transforms.CenterCrop(TARGET_CROP),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def load_model(device: torch.device) -> nn.Module:
    """Load the pretrained ResNet18 backbone (penultimate layer)."""

    # Taking features from the penultimate layer (after global
    # average pooling) yields a compact vector per image that captures high-level
    # content. Freezing ensures we don’t accidentally “train” during extraction.

    weights = models.ResNet18_Weights.IMAGENET1K_V1
    backbone = models.resnet18(weights=weights)
    backbone.eval()
    for param in backbone.parameters():
        param.requires_grad_(False)

    # Penultimate features are obtained after global average pooling
    feature_extractor = nn.Sequential(*list(backbone.children())[:-1])
    feature_extractor.eval()
    feature_extractor.to(device)
    return feature_extractor


from typing import Callable


def preprocess_image(path: Path, transform: Callable[[Image.Image], torch.Tensor]) -> torch.Tensor:
    """Load and preprocess an image, replicating grayscale to RGB."""

    # MRI often comes grayscale; converting to RGB by channel
    # replication satisfies the backbone’s 3-channel input requirement.

    with Image.open(path) as img:
        image_rgb = img.convert("RGB")
        tensor = transform(image_rgb)
    return tensor


def batched(iterable: Sequence, batch_size: int) -> Iterable[Sequence]:
    """Yield successive batches from a sequence."""

    for start in range(0, len(iterable), batch_size):
        end = min(start + batch_size, len(iterable))
        yield iterable[start:end]


def extract_embeddings(
    records: List[ImageRecord],
    device: torch.device,
    batch_size: int = BATCH_SIZE,
) -> ExtractionResults:
    """Run feature extraction for the provided image records."""

    # We handle decode failures explicitly and continue. Silent
    # drops make audits confusing; explicit logs plus a failure list keep the
    # pipeline robust and transparent for students.

    transform = build_transform()
    model = load_model(device)

    embeddings: List[np.ndarray] = []
    kept_records: List[ImageRecord] = []
    failures: List[Path] = []
    per_file_times: List[float] = []

    logging.info("Beginning feature extraction over %d records", len(records))

    for batch_records in batched(records, batch_size):
        batch_tensors: List[torch.Tensor] = []
        successful_records: List[ImageRecord] = []
        batch_start = time.perf_counter()
        for record in batch_records:
            try:
                tensor = preprocess_image(record.absolute_path, transform)
                batch_tensors.append(tensor)
                successful_records.append(record)
            except (UnidentifiedImageError, OSError) as exc:
                logging.error("Failed to decode %s: %s", record.absolute_path, exc)
                failures.append(record.absolute_path)
                continue

        if not batch_tensors:
            continue

        batch_tensor = torch.stack(batch_tensors).to(device)
        with torch.no_grad():
            features = model(batch_tensor)
        # features shape: (B, 2048, 1, 1)
        features = torch.flatten(features, 1)
        embeddings.append(features.cpu().numpy())
        kept_records.extend(successful_records)

        batch_duration = time.perf_counter() - batch_start
        if successful_records:
            per_image = batch_duration / len(successful_records)
            per_file_times.extend([per_image] * len(successful_records))

    if not embeddings:
        raise RuntimeError("No embeddings were generated; all images failed to decode?")

    embedding_matrix = np.concatenate(embeddings, axis=0)
    logging.info("Computed embeddings with shape %s", embedding_matrix.shape)

    return ExtractionResults(
        embeddings=embedding_matrix,
        records=kept_records,
        failures=failures,
        per_file_times=per_file_times,
    )


def compute_dataset_digest(records: Sequence[ImageRecord]) -> str:
    """Compute a deterministic digest of the dataset contents."""

    # A content digest (paths + sizes + mtimes) is a cheap way
    # to detect if the dataset changed between runs. This supports reproducible
    # labs where multiple students share data.

    import hashlib

    hasher = hashlib.sha256()
    for record in sorted(records, key=lambda r: str(r.relative_path)):
        stat = record.absolute_path.stat()
        hasher.update(str(record.relative_path).encode("utf-8"))
        hasher.update(str(stat.st_size).encode("utf-8"))
        hasher.update(str(int(stat.st_mtime)).encode("utf-8"))
    return hasher.hexdigest()


def run_sanity_checks(embeddings: np.ndarray) -> Dict[str, float]:
    """Perform basic integrity checks on the embedding matrix."""

    if np.isnan(embeddings).any():
        raise ValueError("Embedding matrix contains NaN values")
    if np.isinf(embeddings).any():
        raise ValueError("Embedding matrix contains inf values")

    stats = {
        "num_vectors": int(embeddings.shape[0]),
        "dimension": int(embeddings.shape[1]),
        "mean_abs_mean": float(np.abs(embeddings.mean(axis=0)).mean()),
        "mean_std": float(embeddings.std(axis=0).mean()),
    }

    logging.info(
        "Embedding stats — vectors: %d, dim: %d, mean(|mean|): %.5f, mean(std): %.5f",
        stats["num_vectors"],
        stats["dimension"],
        stats["mean_abs_mean"],
        stats["mean_std"],
    )
    return stats


def nearest_neighbor_probe(
    embeddings: np.ndarray,
    records: Sequence[ImageRecord],
    sample_size: int = NEIGHBOR_SAMPLE,
    seed: int = RNG_SEED,
) -> List[Dict[str, object]]:
    """Compute a simple nearest-neighbor spot check on a subset."""

    # This qualitative check builds intuition: similar images
    # should be nearest neighbors in the embedding space if the features are
    # meaningful. We normalise to cosine similarity for scale invariance.

    if embeddings.shape[0] < 2:
        return []

    rng = np.random.default_rng(seed)
    sample_size = min(sample_size, embeddings.shape[0] - 1)
    if sample_size <= 0:
        return []

    sample_indices = rng.choice(embeddings.shape[0], size=sample_size, replace=False)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.clip(norms, a_min=1e-12, a_max=None)
    normalized = embeddings / norms
    neighbor_records: List[Dict[str, object]] = []

    for idx in sample_indices:
        sims = normalized[idx] @ normalized.T
        sims[idx] = -np.inf
        neighbor_idx = int(np.argmax(sims))
        neighbor_records.append(
            {
                "query": str(records[idx].relative_path),
                "neighbor": str(records[neighbor_idx].relative_path),
                "similarity": float(sims[neighbor_idx]),
            }
        )

    logging.info("Nearest-neighbor probe completed for %d samples", len(neighbor_records))
    return neighbor_records


def save_artifacts(
    results: ExtractionResults,
    stats: Dict[str, float],
    neighbor_probe: List[Dict[str, object]],
    data_dir: Path,
    device: torch.device,
) -> None:
    """Persist embeddings, CSV metadata, JSON metadata, and summary notes."""

    # Persisting a CSV that aligns rows to file paths is key for
    # joining with labels, clustering assignments, or error analyses later.

    FEATURE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    NOTE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    np.save(EMBEDDING_ARRAY_PATH, results.embeddings.astype(np.float32))

    rows = []
    for idx, record in enumerate(results.records):
        rows.append(
            {
                "index": idx,
                "path": str(record.relative_path),
                "bucket": record.bucket,
                "label": record.label,
            }
        )
    import pandas as pd

    df = pd.DataFrame(rows)
    df.to_csv(EMBEDDING_CSV_PATH, index=False)

    metadata = {
        "backbone": BACKBONE_NAME,
        "weights": BACKBONE_WEIGHTS,
        "layer": BACKBONE_LAYER,
        "embedding_dimension": int(results.embeddings.shape[1]),
        "input_resize": TARGET_RESIZE,
        "input_crop": TARGET_CROP,
        "normalization_mean": IMAGENET_MEAN,
        "normalization_std": IMAGENET_STD,
        "channel_policy": "PIL RGB conversion (grayscale replicated)",
        "date_utc": datetime.now(timezone.utc).isoformat(),
        "num_images": int(results.embeddings.shape[0]),
        "failed_images": len(results.failures),
        "device": str(device),
        "dataset_dir": str(data_dir),
        "dataset_digest": compute_dataset_digest(results.records),
        "sanity_checks": stats,
        "neighbor_probe": neighbor_probe,
    }

    with METADATA_PATH.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # Build Markdown summary
    failure_section = (
        "None"
        if not results.failures
        else "\n".join(f"- {path}" for path in results.failures)
    )

    mean_latency = float(np.mean(results.per_file_times)) if results.per_file_times else float("nan")
    median_latency = float(np.median(results.per_file_times)) if results.per_file_times else float("nan")

    neighbor_lines = [
        "| Query | Neighbor | Cosine |",
        "| --- | --- | --- |",
    ]
    for item in neighbor_probe:
        neighbor_lines.append(
            f"| {item['query']} | {item['neighbor']} | {item['similarity']:.4f} |"
        )
    neighbor_block = "\n".join(neighbor_lines) if neighbor_probe else "No neighbors computed (insufficient samples)."

    summary = f"""# Feature Extraction Summary

- Backbone: {BACKBONE_NAME} ({BACKBONE_WEIGHTS})
- Layer: global average pooled features ({results.embeddings.shape[1]}-D)
- Input spec: resize {TARGET_RESIZE} → center crop {TARGET_CROP}, RGB conversion, ImageNet normalization
- Batch size: {BATCH_SIZE}
- Device: {device}
- Total images processed: {results.embeddings.shape[0]}
- Failed decodes: {len(results.failures)}
- Mean per-image latency (s): {mean_latency:.4f}
- Median per-image latency (s): {median_latency:.4f}

## Sanity Check Statistics

- Mean of |dimension means|: {stats['mean_abs_mean']:.6f}
- Mean of dimension standard deviations: {stats['mean_std']:.6f}

## Nearest Neighbor Spot Check

{neighbor_block}

## Decode Failures

{failure_section}
"""

    SUMMARY_NOTE_PATH.write_text(summary, encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract CNN embeddings for the MRI dataset")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Root directory containing 'avec_labels' and 'sans_label'",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device to use (default: cuda if available else cpu)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help="Mini-batch size for inference",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(verbose=args.verbose)

    device = torch.device(args.device)

    logging.info("Starting feature extraction on device %s", device)
    records = discover_image_records(args.data_dir)

    start_time = time.perf_counter()
    results = extract_embeddings(records, device=device, batch_size=args.batch_size)
    duration = time.perf_counter() - start_time
    logging.info("Completed embedding extraction in %.2f seconds", duration)

    stats = run_sanity_checks(results.embeddings)
    neighbor_probe = nearest_neighbor_probe(results.embeddings, results.records)
    save_artifacts(results, stats, neighbor_probe, args.data_dir, device)
    logging.info("Artifacts saved to %s", FEATURE_OUTPUT_DIR)


if __name__ == "__main__":
    main()
