"""CLI wrapper for supervised (baseline) training.

Delegates to training.supervised.run_supervised using the shared TrainingConfig
schema from training.common.

Usage:
    python -m src.supervised_training --strong-data-dir <path>
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional, Sequence

from training.common import TrainingConfig
from training.supervised import run_supervised

LOGGER = logging.getLogger(__name__)


def parse_args(args: Optional[Sequence[str]] = None) -> TrainingConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strong-data-dir",
        type=Path,
        required=True,
        help="Path to the strongly labelled dataset (ImageFolder layout).",
    )
    # For supervised, weak data dir isn't used; provide a default unused path
    parser.add_argument(
        "--weak-data-dir",
        type=Path,
        default=Path("unused"),
        help="Unused placeholder for compatibility with TrainingConfig.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--test-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--baseline-epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--early-stopping", type=int, default=3)
    parser.add_argument(
        "--positive-class",
        type=str,
        default="cancer",
        help="Name of the positive class (matching labeled folder name)",
    )
    parser.add_argument(
        "--target-recall",
        type=float,
        default=None,
        help="Optional target recall for validation-based threshold selection (0-1).",
    )
    parser.add_argument("--min-precision", type=float, default=None)
    parser.add_argument("--max-fpr", type=float, default=None)
    parser.add_argument("--f-beta", type=float, default=2.0)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to use: auto (default), cpu, or cuda",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Base directory for experiment artefacts.",
    )

    parsed = parser.parse_args(args=args)
    return TrainingConfig(
        strong_data_dir=parsed.strong_data_dir,
        weak_data_dir=parsed.weak_data_dir,
        batch_size=parsed.batch_size,
        val_split=parsed.val_split,
        test_split=parsed.test_split,
        seed=parsed.seed,
        image_size=parsed.image_size,
        num_workers=parsed.num_workers,
        positive_class=parsed.positive_class,
        target_recall=parsed.target_recall,
        min_precision=parsed.min_precision,
        max_fpr=parsed.max_fpr,
        f_beta=parsed.f_beta,
        baseline_epochs=parsed.baseline_epochs,
        weak_pretrain_epochs=0,  # unused
        finetune_epochs=0,       # unused
        pseudo_label_threshold=0.0,  # unused
        learning_rate=parsed.learning_rate,
        weight_decay=parsed.weight_decay,
        early_stopping_patience=parsed.early_stopping,
        device=parsed.device,
        output_dir=parsed.output_dir,
        results_table=parsed.output_dir / "tables/results_comparison.csv",
        baseline_curve_path=parsed.output_dir / "figures/train_curves_baseline.png",
        semi_curve_path=parsed.output_dir / "figures/train_curves_semi.png",
        baseline_confusion_path=parsed.output_dir / "figures/confusion_matrix_baseline.png",
        semi_confusion_path=parsed.output_dir / "figures/confusion_matrix_semi.png",
        roc_curve_path=parsed.output_dir / "figures/roc_curves.png",
        history_path=parsed.output_dir / "notes/training_history.json",
        baseline_checkpoint=parsed.output_dir / "models/baseline_resnet18.pt",
        semi_checkpoint=parsed.output_dir / "models/semi_resnet18.pt",
        unlabeled_cohort_csv=None,
    )


def main(args: Optional[Sequence[str]] = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s:%(name)s:%(message)s",
    )
    config = parse_args(args)
    metrics = run_supervised(config)
    LOGGER.info("Baseline training complete. Metrics:\n%s", json.dumps(metrics, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
