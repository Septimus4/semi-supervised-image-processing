"""Semi-supervised training pipeline.

This module implements the semi-supervised workflow (baseline → pseudo-label → fine-tune),
reusing shared utilities from training.common.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import json
import time
import logging

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

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
    UnlabeledImageDataset,
    PseudoLabeledDataset,
    make_balanced_sampler,
    plot_metrics_bars,
)

LOGGER = logging.getLogger(__name__)


def generate_pseudo_labels(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    threshold: float = 0.7,
) -> List[Tuple[str, int, float]]:
    """Generate pseudo labels for weakly labelled data.

    Returns list of (path, predicted_label, confidence).
    """
    model.eval()
    pseudo_samples: List[Tuple[str, int, float]] = []
    with torch.no_grad():
        for images, paths in data_loader:
            images = images.to(device)
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            confidences, predictions = torch.max(probabilities, dim=1)
            for path, prediction, confidence in zip(
                paths, predictions.cpu().numpy(), confidences.cpu().numpy()
            ):
                if confidence >= threshold:
                    pseudo_samples.append((path, int(prediction), float(confidence)))
    LOGGER.info(
        "Generated %d pseudo-labelled samples with threshold %.2f",
        len(pseudo_samples),
        threshold,
    )
    return pseudo_samples


def run_pipeline(config: TrainingConfig) -> Dict[str, Dict[str, float]]:
    """Execute the semi-supervised workflow and write artifacts to outputs/.

    Returns a dict with thresholded metrics for baseline and semi-supervised models.
    """
    set_seed(config.seed)
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if config.device == "auto"
        else torch.device(config.device)
    )
    LOGGER.info("Using device: %s", device)

    transforms_map = build_transforms(config.image_size)
    (
        train_loader,
        val_loader,
        test_loader,
        base_dataset,
        split_indices,
    ) = prepare_dataloaders(
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
    criterion = nn.CrossEntropyLoss()

    # Baseline supervised training
    baseline_model = create_model(num_classes=num_classes, pretrained=True).to(device)
    optimizer = optim.AdamW(
        (p for p in baseline_model.parameters() if p.requires_grad),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=2, factor=0.5
    )

    start_time = time.time()
    baseline_model, baseline_history = train_model(
        baseline_model,
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
    baseline_time = time.time() - start_time

    base_arg_metrics, base_arg_y_true, base_arg_y_pred, base_y_prob, _ = (
        evaluate_model(baseline_model, test_loader, device)
    )

    if config.target_recall is not None:
        (_m_bl_val, y_true_val_bl, _pred_val_bl, y_prob_val_bl, _) = evaluate_model(
            baseline_model, val_loader, device, pos_index=pos_index
        )
        thr_baseline, thr_bl_meta = select_operating_threshold(
            (y_true_val_bl == pos_index).astype(int),
            y_prob_val_bl,
            target_recall=float(config.target_recall),
            min_precision=config.min_precision,
            max_fpr=config.max_fpr,
            f_beta=config.f_beta,
        )
        base_thr_metrics, base_thr_y_true, base_thr_y_pred, base_thr_y_prob, _ = (
            evaluate_model(
                baseline_model, test_loader, device, pos_index=pos_index, threshold=thr_baseline
            )
        )
        base_thr_metrics["threshold"] = float(thr_baseline)
        base_thr_metrics["target_recall"] = float(config.target_recall)
        base_thr_metrics["min_precision"] = (
            None if config.min_precision is None else float(config.min_precision)
        )
        base_thr_metrics["max_fpr"] = (
            None if config.max_fpr is None else float(config.max_fpr)
        )
        base_thr_metrics["threshold_policy"] = thr_bl_meta.get("policy", "unknown")
    else:
        thr_baseline = None
        base_thr_metrics = dict(base_arg_metrics)
        base_thr_y_true, base_thr_y_pred, base_thr_y_prob = (
            base_arg_y_true,
            base_arg_y_pred,
            base_y_prob,
        )
        base_thr_metrics["threshold"] = None
        base_thr_metrics["target_recall"] = None
        base_thr_metrics["min_precision"] = None
        base_thr_metrics["max_fpr"] = None
        base_thr_metrics["threshold_policy"] = "disabled"

    base_thr_metrics["training_time_sec"] = baseline_time

    plot_training_curves(baseline_history, config.baseline_curve_path, "Baseline")

    # Pseudo-labeling on unlabeled pool
    unlabeled_dataset = UnlabeledImageDataset(
        config.weak_data_dir, transform=transforms_map["eval"]
    )
    if config.unlabeled_cohort_csv is not None:
        cohort_path = Path(config.unlabeled_cohort_csv)
        if not cohort_path.exists():
            raise FileNotFoundError(f"Cohort CSV not found: {cohort_path}")
        cohort_df = pd.read_csv(cohort_path)
        if "path" not in cohort_df.columns:
            raise ValueError("Cohort CSV must contain a 'path' column")
        allowed: set[str] = set()
        weak_name = config.weak_data_dir.name
        for p in cohort_df["path"].astype(str).tolist():
            pp = Path(p)
            candidates = set()
            if pp.is_absolute():
                candidates.add(pp.resolve())
            else:
                candidates.add((config.weak_data_dir / pp).resolve())
                parts = pp.parts
                if len(parts) > 1 and parts[0] == weak_name:
                    candidates.add((config.weak_data_dir / Path(*parts[1:])).resolve())
                if len(parts) == 1:
                    candidates.add((config.weak_data_dir / pp.name).resolve())
            for c in candidates:
                allowed.add(str(c))
        before = len(unlabeled_dataset.image_paths)
        unlabeled_dataset.image_paths = [
            Path(p) for p in unlabeled_dataset.image_paths if str(Path(p).resolve()) in allowed
        ]
        after = len(unlabeled_dataset.image_paths)
        LOGGER.info(
            "Filtered unlabeled pool via cohort CSV: %d -> %d images (%d excluded)",
            before,
            after,
            before - after,
        )
        if after == 0:
            raise RuntimeError(
                "Cohort filtering removed all unlabeled images; check the CSV paths match --weak-data-dir."
            )
    unlabeled_loader = DataLoader(
        unlabeled_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    pseudo_samples = generate_pseudo_labels(
        baseline_model, unlabeled_loader, device, config.pseudo_label_threshold
    )

    pseudo_dataset = PseudoLabeledDataset(
        [(path, label) for path, label, _ in pseudo_samples],
        transform=transforms_map["train"],
    )
    if len(pseudo_dataset) == 0:
        raise RuntimeError(
            "No pseudo-labelled samples were generated. Try lowering the threshold."
        )

    pseudo_targets = [label for _, label, _ in pseudo_samples]
    pseudo_sampler = make_balanced_sampler(pseudo_targets)
    pseudo_loader = DataLoader(
        pseudo_dataset,
        batch_size=config.batch_size,
        sampler=pseudo_sampler,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    semi_model = create_model(num_classes=num_classes, pretrained=True).to(device)
    for name, param in semi_model.named_parameters():
        if not name.startswith("fc"):
            param.requires_grad = False

    optimizer_pretrain = optim.AdamW(
        (p for p in semi_model.parameters() if p.requires_grad),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    pretrain_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_pretrain, mode="min", patience=2, factor=0.5
    )

    start_time = time.time()
    semi_model, pretrain_history = train_model(
        semi_model,
        pseudo_loader,
        val_loader,
        criterion,
        optimizer_pretrain,
        device,
        scheduler=pretrain_scheduler,
        num_epochs=config.weak_pretrain_epochs,
        early_stopping_patience=config.early_stopping_patience,
    )

    # Fine-tuning on strong labels
    for param in semi_model.parameters():
        param.requires_grad = True

    optimizer_finetune = optim.AdamW(
        semi_model.parameters(),
        lr=config.learning_rate / 2,
        weight_decay=config.weight_decay,
    )
    finetune_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_finetune, mode="min", patience=2, factor=0.5
    )

    semi_model, finetune_history = train_model(
        semi_model,
        train_loader,
        val_loader,
        criterion,
        optimizer_finetune,
        device,
        scheduler=finetune_scheduler,
        num_epochs=config.finetune_epochs,
        early_stopping_patience=config.early_stopping_patience,
        model_path=config.semi_checkpoint,
    )
    semi_time = time.time() - start_time

    # Evaluate semi-supervised model
    semi_arg_metrics, semi_arg_y_true, semi_arg_y_pred, semi_y_prob, _ = evaluate_model(
        semi_model, test_loader, device
    )

    if config.target_recall is not None:
        (_m_se_val, y_true_val_se, _pred_val_se, y_prob_val_se, _) = evaluate_model(
            semi_model, val_loader, device, pos_index=pos_index
        )
        thr_semi, thr_se_meta = select_operating_threshold(
            (y_true_val_se == pos_index).astype(int),
            y_prob_val_se,
            target_recall=float(config.target_recall),
            min_precision=config.min_precision,
            max_fpr=config.max_fpr,
            f_beta=config.f_beta,
        )
        semi_thr_metrics, semi_thr_y_true, semi_thr_y_pred, semi_thr_y_prob, _ = evaluate_model(
            semi_model, test_loader, device, pos_index=pos_index, threshold=thr_semi
        )
        semi_thr_metrics["threshold"] = float(thr_semi)
        semi_thr_metrics["target_recall"] = float(config.target_recall)
        semi_thr_metrics["min_precision"] = (
            None if config.min_precision is None else float(config.min_precision)
        )
        semi_thr_metrics["max_fpr"] = (
            None if config.max_fpr is None else float(config.max_fpr)
        )
        semi_thr_metrics["threshold_policy"] = thr_se_meta.get("policy", "unknown")
    else:
        thr_semi = None
        semi_thr_metrics = dict(semi_arg_metrics)
        semi_thr_y_true, semi_thr_y_pred, semi_thr_y_prob = (
            semi_arg_y_true,
            semi_arg_y_pred,
            semi_y_prob,
        )
        semi_thr_metrics["threshold"] = None
        semi_thr_metrics["target_recall"] = None
        semi_thr_metrics["min_precision"] = None
        semi_thr_metrics["max_fpr"] = None
        semi_thr_metrics["threshold_policy"] = "disabled"

    semi_thr_metrics["training_time_sec"] = semi_time

    # Persist histories and figures
    history_payload = {
        "baseline": baseline_history,
        "semi_pretrain": pretrain_history,
        "semi_finetune": finetune_history,
        "splits": {k: v.tolist() for k, v in split_indices.items()},
        "pseudo_label_count": len(pseudo_samples),
    }
    config.history_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config.history_path, "w", encoding="utf-8") as fp:
        json.dump(history_payload, fp, indent=2)

    plot_training_curves(
        {
            "train_loss": pretrain_history["train_loss"] + finetune_history["train_loss"],
            "val_loss": pretrain_history["val_loss"] + finetune_history["val_loss"],
            "train_acc": pretrain_history["train_acc"] + finetune_history["train_acc"],
            "val_acc": pretrain_history["val_acc"] + finetune_history["val_acc"],
            "train_f1": pretrain_history["train_f1"] + finetune_history["train_f1"],
            "val_f1": pretrain_history["val_f1"] + finetune_history["val_f1"],
        },
        config.semi_curve_path,
        "Semi-supervised",
    )

    plot_confusion_matrix(base_arg_y_true, base_arg_y_pred, base_dataset.classes, config.baseline_confusion_path)
    plot_confusion_matrix(base_thr_y_true, base_thr_y_pred, base_dataset.classes, Path("outputs/figures/confusion_matrix_baseline_thresholded.png"))
    plot_confusion_matrix(semi_arg_y_true, semi_arg_y_pred, base_dataset.classes, config.semi_confusion_path)
    plot_confusion_matrix(semi_thr_y_true, semi_thr_y_pred, base_dataset.classes, Path("outputs/figures/confusion_matrix_semi_thresholded.png"))

    baseline_y_true_bin = (base_thr_y_true == pos_index).astype(int)
    semi_y_true_bin = (semi_thr_y_true == pos_index).astype(int)
    plot_roc_curves(
        {
            "Baseline": (baseline_y_true_bin, base_thr_y_prob),
            "Semi-supervised": (semi_y_true_bin, semi_thr_y_prob),
        },
        config.roc_curve_path,
    )

    plot_pr_curves(
        {
            "Baseline": (baseline_y_true_bin, base_thr_y_prob),
            "Semi-supervised": (semi_y_true_bin, semi_thr_y_prob),
        },
        Path("outputs/figures/pr_curves.png"),
    )

    detailed_rows: Dict[str, Dict[str, float]] = {}
    detailed_rows["baseline_argmax"] = compute_binary_confusion_metrics(
        base_arg_y_true, base_arg_y_pred, pos_index
    ) | {"threshold": None, "target_recall": None, "training_time_sec": base_arg_metrics.get("training_time_sec", baseline_time)}
    detailed_rows["baseline_thresholded"] = compute_binary_confusion_metrics(
        base_thr_y_true, base_thr_y_pred, pos_index
    ) | {
        "threshold": (None if thr_baseline is None else float(thr_baseline)),
        "target_recall": (None if config.target_recall is None else float(config.target_recall)),
        "training_time_sec": base_thr_metrics.get("training_time_sec", baseline_time),
        "min_precision": base_thr_metrics.get("min_precision"),
        "max_fpr": base_thr_metrics.get("max_fpr"),
    }
    detailed_rows["semi_argmax"] = compute_binary_confusion_metrics(
        semi_arg_y_true, semi_arg_y_pred, pos_index
    ) | {"threshold": None, "target_recall": None, "training_time_sec": semi_arg_metrics.get("training_time_sec", semi_time)}
    detailed_rows["semi_thresholded"] = compute_binary_confusion_metrics(
        semi_thr_y_true, semi_thr_y_pred, pos_index
    ) | {
        "threshold": (None if thr_semi is None else float(thr_semi)),
        "target_recall": (None if config.target_recall is None else float(config.target_recall)),
        "training_time_sec": semi_thr_metrics.get("training_time_sec", semi_time),
        "min_precision": semi_thr_metrics.get("min_precision"),
        "max_fpr": semi_thr_metrics.get("max_fpr"),
    }

    detailed_df = pd.DataFrame.from_dict(detailed_rows, orient="index")
    Path("outputs/tables").mkdir(parents=True, exist_ok=True)
    detailed_df.to_csv(Path("outputs/tables/results_comparison_detailed.csv"))

    plot_metrics_bars(
        detailed_rows,
        Path("outputs/figures/metrics_comparison.png"),
        keys=["TPR", "FPR", "TNR", "precision", "accuracy"],
    )

    results_df = pd.DataFrame.from_dict(
        {
            "baseline_thresholded": base_thr_metrics,
            "semi_thresholded": semi_thr_metrics,
        },
        orient="index",
    )
    config.results_table.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(config.results_table)

    # Operating point manifest
    try:
        op_payload = {
            "model": "semi_supervised_resnet18",
            "checkpoint": str(config.semi_checkpoint),
            "positive_class": config.positive_class,
            "threshold": (None if "threshold" not in semi_thr_metrics else semi_thr_metrics.get("threshold")),
            "policy": semi_thr_metrics.get("threshold_policy"),
            "target_recall": config.target_recall,
            "min_precision": config.min_precision,
            "max_fpr": config.max_fpr,
            "seed": config.seed,
        }
        config.operating_point_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config.operating_point_path, "w", encoding="utf-8") as fp:
            json.dump(op_payload, fp, indent=2)
    except Exception as exc:
        LOGGER.warning("Failed to write operating_point.json: %s", exc)

    # Triage CSV for unlabeled pool
    try:
        triage_rows: List[Dict[str, Any]] = []
        triage_thr = semi_thr_metrics.get("threshold")
        if triage_thr is not None:
            triage_loader = DataLoader(
                unlabeled_dataset,
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=config.num_workers,
                pin_memory=torch.cuda.is_available(),
            )
            semi_model.eval()
            with torch.no_grad():
                for images, paths in triage_loader:
                    images = images.to(device)
                    outputs = semi_model(images)
                    probs_full = torch.softmax(outputs, dim=1)
                    pos_probs = probs_full[:, pos_index].detach().cpu().numpy().tolist()
                    for pth, pr in zip(paths, pos_probs):
                        triage_rows.append(
                            {
                                "path": str(pth),
                                "prob_positive": float(pr),
                                "flagged": bool(pr >= float(triage_thr)),
                            }
                        )
            df_triage = pd.DataFrame(triage_rows)
            config.triage_csv_path.parent.mkdir(parents=True, exist_ok=True)
            df_triage.to_csv(config.triage_csv_path, index=False)
            LOGGER.info(
                "Wrote triage CSV with %d rows (%d flagged) to %s",
                len(df_triage),
                int(df_triage["flagged"].sum()) if not df_triage.empty else 0,
                config.triage_csv_path,
            )
        else:
            LOGGER.info("Skipping triage CSV: no threshold selected (thresholding disabled)")
    except Exception as exc:
        LOGGER.warning("Failed to write triage CSV: %s", exc)

    return {
        "baseline_thresholded": base_thr_metrics,
        "semi_thresholded": semi_thr_metrics,
    }
