"""Supervised (baseline) training pipeline.

This module trains and evaluates a supervised baseline model using utilities
from training.common. It is intended for comparison against the semi-supervised
variant and for standalone supervised experiments.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence
import logging

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from .common import (
    TrainingConfig,
    set_seed,
    build_transforms,
    prepare_dataloaders,
    create_model,
    train_model,
    evaluate_model,
    select_operating_threshold,
    plot_training_curves,
    plot_confusion_matrix,
    plot_roc_curves,
    plot_pr_curves,
    compute_binary_confusion_metrics,
)

LOGGER = logging.getLogger(__name__)


def run_supervised(config: TrainingConfig) -> Dict[str, Dict[str, float]]:
    """Train and evaluate a baseline supervised model.

    Returns a dict with argmax and (optionally) thresholded metrics.
    """
    set_seed(config.seed)
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if config.device == "auto"
        else torch.device(config.device)
    )
    LOGGER.info("Using device: %s", device)

    transforms_map = build_transforms(config.image_size)
    train_loader, val_loader, test_loader, base_dataset, _split_indices = prepare_dataloaders(
        config.strong_data_dir,
        transforms_map,
        config.batch_size,
        config.val_split,
        config.test_split,
        config.seed,
        config.num_workers,
    )

    num_classes = len(base_dataset.classes)
    if config.positive_class not in base_dataset.class_to_idx:
        raise ValueError(
            f"Positive class '{config.positive_class}' not found in dataset classes: {base_dataset.classes}"
        )
    pos_index = int(base_dataset.class_to_idx[config.positive_class])

    model = create_model(num_classes=num_classes, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=2, factor=0.5
    )

    model, history = train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        device,
        scheduler=scheduler,
        num_epochs=config.baseline_epochs,
        early_stopping_patience=config.early_stopping_patience,
        model_path=config.baseline_checkpoint,
    )

    # Argmax metrics
    arg_metrics, arg_y_true, arg_y_pred, y_prob_test, _ = evaluate_model(model, test_loader, device)

    # Thresholded metrics (optional)
    if config.target_recall is not None:
        (_m_val, y_true_val, _pred_val, y_prob_val, _) = evaluate_model(
            model, val_loader, device, pos_index=pos_index
        )
        thr, thr_meta = select_operating_threshold(
            (y_true_val == pos_index).astype(int),
            y_prob_val,
            target_recall=float(config.target_recall),
            min_precision=config.min_precision,
            max_fpr=config.max_fpr,
            f_beta=config.f_beta,
        )
        thr_metrics, thr_y_true, thr_y_pred, thr_y_prob, _ = evaluate_model(
            model, test_loader, device, pos_index=pos_index, threshold=thr
        )
        thr_metrics["threshold"] = float(thr)
        thr_metrics["target_recall"] = float(config.target_recall)
        thr_metrics["min_precision"] = (
            None if config.min_precision is None else float(config.min_precision)
        )
        thr_metrics["max_fpr"] = (
            None if config.max_fpr is None else float(config.max_fpr)
        )
        thr_metrics["threshold_policy"] = thr_meta.get("policy", "unknown")
    else:
        thr = None
        thr_metrics = dict(arg_metrics)
        thr_y_true, thr_y_pred, thr_y_prob = arg_y_true, arg_y_pred, y_prob_test
        thr_metrics["threshold"] = None
        thr_metrics["target_recall"] = None
        thr_metrics["min_precision"] = None
        thr_metrics["max_fpr"] = None
        thr_metrics["threshold_policy"] = "disabled"

    # Plots
    plot_training_curves(history, config.baseline_curve_path, "Baseline")
    plot_confusion_matrix(arg_y_true, arg_y_pred, base_dataset.classes, config.baseline_confusion_path)

    baseline_y_true_bin = (thr_y_true == pos_index).astype(int)
    plot_roc_curves({"Baseline": (baseline_y_true_bin, thr_y_prob)}, config.roc_curve_path)
    plot_pr_curves({"Baseline": (baseline_y_true_bin, thr_y_prob)}, Path("outputs/figures/pr_curves_baseline.png"))

    # Save summary table for baseline only
    results_df = pd.DataFrame.from_dict({"baseline_thresholded": thr_metrics}, orient="index")
    (config.results_table.parent).mkdir(parents=True, exist_ok=True)
    results_df.to_csv(config.results_table)

    return {"baseline_thresholded": thr_metrics, "baseline_argmax": arg_metrics}
