"""Semi-supervised and supervised training pipeline for brain MRI classification.

This module provides utilities to train a baseline supervised model and a
semi-supervised variant that leverages weakly labelled data. The goal is to
support experimentation aligned with Task 4 of the project brief.

The code is organised around a set of helper functions for data preparation,
model training, evaluation, and visualisation. A command-line interface is
provided so the pipeline can be executed end-to-end from the terminal:

```
python -m src.semi_supervised_training --strong-data-dir <path> --weak-data-dir <path>
```

Notes (why this structure?):
- We implement both models (supervised baseline and semi-supervised) in one script so
    students can compare apples-to-apples on the exact same split and metrics. This reduces
    confounds from different random seeds or preprocessing.
- We save intermediate artifacts (splits, training curves, metrics tables) to make results
    reproducible and to support “what-if” analysis (e.g., different thresholds or hyperparameters).
- We explicitly separate data prep, training, and evaluation to highlight the lifecycle of
    an ML experiment and to encourage modular thinking.

See the README or ``notes/training_report.md`` for additional guidance.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import logging
import math
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    precision_recall_curve,
    average_precision_score,
    precision_recall_fscore_support,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import datasets, models, transforms
from PIL import Image
# (matplotlib already imported as plt above)


LOGGER = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Ensure reproducible behaviour across libraries."""

    # Determinism is crucial for debugging and fair comparisons. We fix
    # common random sources so that splits and curves are comparable.
    # This can cost a bit of speed (e.g., disabling cuDNN benchmarking)
    # but greatly improves clarity.

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_transforms(image_size: int = 224) -> Dict[str, transforms.Compose]:
    """Construct train and evaluation transforms."""

    # We use light augmentation on train to improve generalisation and
    # deterministic resizing/normalisation to match the backbone’s
    # expectations.
    # Why 224×224 (with a 256→224 resize-crop)?
    # - Torchvision’s ResNet-18 is pretrained on ImageNet at 224×224.
    #   Matching this resolution keeps receptive fields and statistics
    #   aligned, which improves transfer.
    # - The common 256→224 pipeline performs a modest downscale then a
    #   center crop to reduce border artifacts and standardize content.
    # - 224 strikes a balance between detail retention and VRAM/throughput.
    #   If your images have very fine structures, you can increase size at
    #   the cost of memory and speed; ensure to adjust the feature extractor
    #   and data pipelines consistently.

    normalization = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            normalization,
        ]
    )

    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalization,
        ]
    )

    return {"train": train_transform, "eval": eval_transform}


class TransformSubset(Dataset):
    """Apply a transform to a subset of a base dataset."""

    # Rather than duplicating files, we wrap an ImageFolder
    # and expose only the indices for train/val/test. This is memory- and
    # disk-efficient and keeps class mapping consistent.

    def __init__(
        self,
        dataset: datasets.ImageFolder,
        indices: Sequence[int],
        transform: Optional[transforms.Compose] = None,
        return_paths: bool = False,
    ) -> None:
        self.dataset = dataset
        self.indices = list(indices)
        self.transform = transform
        self.return_paths = return_paths

    def __len__(self) -> int:  # pragma: no cover - simple delegation
        return len(self.indices)

    def __getitem__(self, idx: int):  # pragma: no cover - exercised via loaders
        image, label = self.dataset[self.indices[idx]]
        if self.transform is not None:
            image = self.transform(image)

        if self.return_paths:
            path = self.dataset.samples[self.indices[idx]][0]
            return image, label, path
        return image, label


class UnlabeledImageDataset(Dataset):
    """Dataset returning images (and paths) from an unlabeled directory."""

    # The weak pool has no labels on disk. We return paths
    # alongside tensors so we can later attach pseudo-labels and audit which
    # files were used.

    def __init__(
        self, root_dir: Path, transform: Optional[transforms.Compose] = None
    ) -> None:
        self.root_dir = Path(root_dir)
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Unlabeled directory not found: {self.root_dir}")

        self.image_paths = sorted(
            [
                path
                for path in self.root_dir.iterdir()
                if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
            ]
        )
        self.transform = transform

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.image_paths)

    def __getitem__(self, idx: int):  # pragma: no cover - used in loaders
        path = self.image_paths[idx]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, str(path)


class PseudoLabeledDataset(Dataset):
    """Dataset that pairs image paths with pseudo labels."""

    # Pseudo-labelling converts the weak pool into a
    # supervised dataset by trusting the baseline model’s confident
    # predictions. Confidence thresholding trades coverage for label quality.

    def __init__(
        self,
        samples: Sequence[Tuple[str, int]],
        transform: Optional[transforms.Compose] = None,
    ) -> None:
        self.samples = list(samples)
        self.transform = transform

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.samples)

    def __getitem__(self, idx: int):  # pragma: no cover - used in loaders
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def stratified_split(
    targets: Sequence[int],
    val_size: float,
    test_size: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stratified train/val/test split indices."""

    # Stratification maintains class balance in each split,
    # which is especially important in medical settings where classes can be
    # imbalanced. We split once and persist indices for reproducibility.

    indices = np.arange(len(targets))
    train_indices, temp_indices, _, temp_targets = train_test_split(
        indices,
        targets,
        test_size=val_size + test_size,
        random_state=seed,
        stratify=targets,
    )

    relative_test_size = test_size / (val_size + test_size)
    val_indices, test_indices = train_test_split(
        temp_indices,
        test_size=relative_test_size,
        random_state=seed,
        stratify=temp_targets,
    )

    return (
        np.array(train_indices),
        np.array(val_indices),
        np.array(test_indices),
    )


def make_balanced_sampler(labels: Sequence[int]) -> WeightedRandomSampler:
    """Create a class-balanced sampler for binary classification."""

    # Even if the dataset is roughly balanced overall, a
    # specific train split might not be. A weighted sampler helps avoid a
    # model that over-predicts the majority class by up-weighting rare labels.

    label_array = np.array(labels)
    class_sample_count = np.bincount(label_array)
    if len(np.nonzero(class_sample_count)[0]) < 2:
        LOGGER.warning(
            "Only one class present in labels; using uniform sampling instead of balancing."
        )
        # WeightedRandomSampler expects a Sequence[float]; provide a Python list
        return WeightedRandomSampler(
            weights=[1.0] * int(len(label_array)),
            num_samples=int(len(label_array)),
            replacement=True,
        )

    weight_per_class = 1.0 / class_sample_count
    samples_weight = weight_per_class[label_array].astype(float)
    return WeightedRandomSampler(
        weights=samples_weight.tolist(),
        num_samples=int(len(samples_weight)),
        replacement=True,
    )


def prepare_dataloaders(
    strong_data_dir: Path,
    transforms_map: Dict[str, transforms.Compose],
    batch_size: int,
    val_split: float,
    test_split: float,
    seed: int,
    num_workers: int = 2,
) -> Tuple[DataLoader, DataLoader, DataLoader, datasets.ImageFolder, Dict[str, np.ndarray]]:
    """Create train, validation, and test dataloaders."""

    # We materialise three loaders with different transforms
    # (augmentation for train, deterministic for eval). We also include paths
    # in val/test batches so evaluation can save per-sample artifacts.

    base_dataset = datasets.ImageFolder(strong_data_dir, transform=None)
    targets = np.array(base_dataset.targets)
    train_idx, val_idx, test_idx = stratified_split(targets.tolist(), val_split, test_split, seed)

    split_indices = {"train": train_idx, "val": val_idx, "test": test_idx}

    train_dataset = TransformSubset(
    base_dataset, list(train_idx), transform=transforms_map["train"]
    )
    val_dataset = TransformSubset(
    base_dataset, list(val_idx), transform=transforms_map["eval"], return_paths=True
    )
    test_dataset = TransformSubset(
    base_dataset, list(test_idx), transform=transforms_map["eval"], return_paths=True
    )

    train_targets = targets[train_idx]
    sampler = make_balanced_sampler(train_targets.tolist())

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, test_loader, base_dataset, split_indices


def create_model(num_classes: int, pretrained: bool = True) -> nn.Module:
    """Initialise a ResNet-18 backbone with a new classification head."""

    # ResNet-18 is a good backbone for this scale: small enough to
    # train quickly, but strong when initialised with ImageNet weights. We
    # replace the final FC layer to match our dataset classes.

    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    scheduler: Optional[optim.lr_scheduler._LRScheduler | optim.lr_scheduler.ReduceLROnPlateau] = None,
    num_epochs: int = 10,
    early_stopping_patience: int = 3,
    model_path: Optional[Path] = None,
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    """Generic training loop with early stopping."""

    # This loop logs both accuracy and F1 because medical
    # datasets care about positive-class performance. Early stopping and an
    # LR scheduler prevent overfitting and stabilise training for students.

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "train_f1": [],
        "val_f1": [],
    }

    best_state = copy.deepcopy(model.state_dict())
    best_val_loss = math.inf
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        train_losses: List[float] = []
        y_true_train: List[int] = []
        y_pred_train: List[int] = []

        for inputs, labels in train_loader:
            # Standard supervised step: forward → loss → backward → step
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            predictions = outputs.argmax(dim=1)
            y_true_train.extend(labels.cpu().numpy().tolist())
            y_pred_train.extend(predictions.cpu().numpy().tolist())

        train_loss = float(np.mean(train_losses)) if train_losses else 0.0
        train_acc, train_f1 = compute_accuracy_f1(y_true_train, y_pred_train)

        val_loss, val_acc, val_f1 = evaluate_on_loader(
            model, val_loader, criterion, device
        )

        if scheduler is not None:
            # ReduceLROnPlateau expects the validation metric; standard schedulers do not.
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["train_f1"].append(train_f1)
        history["val_f1"].append(val_f1)

        LOGGER.info(
            "Epoch %d/%d - train loss %.4f acc %.3f f1 %.3f | val loss %.4f acc %.3f f1 %.3f",
            epoch + 1,
            num_epochs,
            train_loss,
            train_acc,
            train_f1,
            val_loss,
            val_acc,
            val_f1,
        )

        if val_loss < best_val_loss:
            # Keep the best weights by validation loss; this is stricter than
            # just taking the last epoch and avoids regressions late in training.
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            if model_path is not None:
                model_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(best_state, model_path)
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                LOGGER.info("Early stopping triggered at epoch %d", epoch + 1)
                break

    model.load_state_dict(best_state)
    return model, history


def compute_accuracy_f1(
    y_true: Sequence[int], y_pred: Sequence[int]
) -> Tuple[float, float]:
    """Return accuracy and F1 score."""

    if len(y_true) == 0:
        return 0.0, 0.0
    accuracy = accuracy_score(y_true, y_pred)
    _, _, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    return float(accuracy), float(f1)


def evaluate_on_loader(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, float]:
    """Evaluate model returning loss, accuracy, and F1."""

    model.eval()
    losses: List[float] = []
    y_true: List[int] = []
    y_pred: List[int] = []

    with torch.no_grad():
        for batch in data_loader:
            inputs, labels = batch[:2]
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            losses.append(loss.item())

            predictions = outputs.argmax(dim=1)
            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(predictions.cpu().numpy().tolist())

    avg_loss = float(np.mean(losses)) if losses else 0.0
    acc, f1 = compute_accuracy_f1(y_true, y_pred)
    return avg_loss, acc, f1


def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    pos_index: Optional[int] = None,
    threshold: Optional[float] = None,
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Evaluate the model and capture predictions for reporting.

        If ``pos_index`` is provided, probabilities correspond to that positive class.
        If ``threshold`` is provided and the problem is binary, predictions are made by
        thresholding the positive-class probability instead of argmax.

        Why threshold at all?
        - Argmax is optimal for accuracy but can miss positives when we care about
            recall (sensitivity). Thresholding lets us “slide” the operating point to
            catch more positives at the expense of more false alarms.
    """

    model.eval()
    y_true: List[int] = []
    y_pred: List[int] = []
    y_prob: List[float] = []
    sample_paths: List[str] = []

    with torch.no_grad():
        for batch in data_loader:
            inputs, labels = batch[:2]
            extras = batch[2:] if len(batch) > 2 else []
            # Ensure paths is a list[str] for typing correctness
            paths = extras[0] if extras else ["" for _ in range(len(labels))]

            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            probs_full = torch.softmax(outputs, dim=1)
            if pos_index is None:
                pos_col = 1 if probs_full.shape[1] > 1 else 0
            else:
                pos_col = pos_index
            probabilities = probs_full[:, pos_col]
            if threshold is None:
                predictions = outputs.argmax(dim=1)
            else:
                if probs_full.shape[1] == 2:
                    neg_col = 1 - pos_col
                    bin_pred = (probabilities >= threshold).to(torch.long)
                    predictions = torch.where(
                        bin_pred == 1,
                        torch.tensor(pos_col, device=bin_pred.device),
                        torch.tensor(neg_col, device=bin_pred.device),
                    )
                else:
                    # Fallback to argmax for multi-class
                    predictions = outputs.argmax(dim=1)

            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(predictions.cpu().numpy().tolist())
            y_prob.extend(probabilities.cpu().numpy().tolist())
            sample_paths.extend([str(p) for p in list(paths)])

    if pos_index is not None:
        y_true_bin = (np.array(y_true) == pos_index).astype(int)
        y_pred_bin = (np.array(y_pred) == pos_index).astype(int)
        accuracy = accuracy_score(y_true_bin, y_pred_bin)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true_bin, y_pred_bin, average="binary", zero_division=0
        )
    else:
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )

    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }

    return metrics, np.array(y_true), np.array(y_pred), np.array(y_prob), sample_paths


def find_threshold_for_target_recall(
    y_true_bin: np.ndarray, y_prob: np.ndarray, target_recall: float
) -> float:
    """Return the highest threshold achieving at least target_recall.

    If positives are absent, returns 0.5. If no threshold reaches target recall,
    returns the minimum threshold (close to 0), effectively maximizing recall.
    """
    # We iterate thresholds high→low so we choose the largest
    # threshold that still meets the recall target. Larger thresholds reduce
    # false positives, so this selection is “safest” among those that satisfy
    # the recall constraint.
    if y_true_bin.sum() == 0:
        return 0.5
    thresholds = np.unique(np.concatenate(([0.0], y_prob)))
    thresholds.sort()
    best_thr = thresholds[0]
    # Iterate from high to low to pick the largest threshold that still meets recall
    for thr in thresholds[::-1]:
        y_pred = (y_prob >= thr).astype(int)
        _, recall, _, _ = precision_recall_fscore_support(
            y_true_bin, y_pred, average="binary", zero_division=0
        )
        if recall >= target_recall:
            best_thr = float(thr)
            break
    return float(best_thr)


def select_operating_threshold(
    y_true_bin: np.ndarray,
    y_prob: np.ndarray,
    target_recall: float,
    min_precision: Optional[float] = None,
    max_fpr: Optional[float] = None,
    f_beta: float = 2.0,
) -> Tuple[float, Dict[str, Any]]:
    """Select a threshold prioritizing recall with optional precision/FPR constraints.

    Strategy:
    1) Filter thresholds that achieve recall >= target_recall.
       - If ``min_precision`` is provided, also require precision >= min_precision.
       - If ``max_fpr`` is provided, also require FPR <= max_fpr.
       Among feasible thresholds, pick the largest (to minimize false positives).
    2) If no threshold satisfies constraints, fall back to maximizing F-beta
       (beta>1 emphasizes recall). Break ties by choosing the largest threshold.
    3) As a last resort, choose the highest threshold that achieves target_recall
       ignoring other constraints; if that also fails, choose the minimum threshold.

    Returns the chosen threshold and a small metrics dict describing the decision.
    """
    if y_true_bin.sum() == 0:
        return 0.5, {"policy": "no_positives", "recall": 0.0, "precision": 0.0, "fpr": 0.0}

    thresholds = np.unique(np.concatenate(([0.0], y_prob, [1.0])))
    thresholds.sort()

    def stats_at(thr: float) -> Tuple[float, float, float, float, float, float]:
        y_pred = (y_prob >= thr).astype(int)
        tp = float(((y_true_bin == 1) & (y_pred == 1)).sum())
        tn = float(((y_true_bin == 0) & (y_pred == 0)).sum())
        fp = float(((y_true_bin == 0) & (y_pred == 1)).sum())
        fn = float(((y_true_bin == 1) & (y_pred == 0)).sum())
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        # F-beta
        if precision + recall > 0:
            beta2 = f_beta * f_beta
            fbeta = (1 + beta2) * (precision * recall) / (beta2 * precision + recall)
        else:
            fbeta = 0.0
        return recall, precision, fpr, tp, fp, fbeta

    # 1) Feasible set under constraints
    feasible: List[Tuple[float, float, float, float]] = []  # (thr, recall, precision, fpr)
    for thr in thresholds:
        recall, precision, fpr, _tp, _fp, _fbeta = stats_at(thr)
        if recall + 1e-12 < target_recall:
            continue
        if min_precision is not None and precision + 1e-12 < min_precision:
            continue
        if max_fpr is not None and fpr - 1e-12 > max_fpr:
            continue
        feasible.append((float(thr), recall, precision, fpr))

    if feasible:
        # Choose the largest threshold to be conservative on positives
        thr, recall, precision, fpr = sorted(feasible, key=lambda x: x[0])[-1]
        return float(thr), {
            "policy": "constrained",
            "recall": float(recall),
            "precision": float(precision),
            "fpr": float(fpr),
        }

    # 2) F-beta fallback
    scored: List[Tuple[float, float, float, float]] = []  # (fbeta, thr, recall, precision)
    for thr in thresholds:
        recall, precision, fpr, _tp, _fp, fbeta = stats_at(thr)
        scored.append((fbeta, float(thr), recall, precision))
    scored.sort(key=lambda x: (x[0], x[1]))  # prefer larger thr on tie by sorting then picking last by thr later
    best_fbeta = max(scored, key=lambda x: (x[0], x[1]))
    fbeta, thr, recall, precision = best_fbeta
    if fbeta > 0:
        _, _, fpr, *_ = stats_at(thr)
        return float(thr), {
            "policy": "fbeta",
            "fbeta": float(fbeta),
            "recall": float(recall),
            "precision": float(precision),
            "fpr": float(fpr),
        }

    # 3) Recall-only fallback then minimum threshold
    recall_only_thr = find_threshold_for_target_recall(y_true_bin, y_prob, target_recall)
    if recall_only_thr is not None:
        r, p, fpr, *_ = stats_at(recall_only_thr)
        return float(recall_only_thr), {
            "policy": "recall_only",
            "recall": float(r),
            "precision": float(p),
            "fpr": float(fpr),
        }
    thr0 = float(thresholds[0])
    r, p, fpr, *_ = stats_at(thr0)
    return thr0, {"policy": "min_threshold", "recall": float(r), "precision": float(p), "fpr": float(fpr)}


def generate_pseudo_labels(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    threshold: float = 0.7,
) -> List[Tuple[str, int, float]]:
    """Generate pseudo labels for weakly labelled data."""

    # The confidence threshold is a knob. Higher values give
    # fewer pseudo-labels (lower coverage) but higher precision; lower values
    # give more data with noisier labels. We record confidences for auditing.

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


def plot_training_curves(
    history: Dict[str, List[float]], output_path: Path, title: str
) -> None:
    """Plot loss and F1 curves for train/validation."""

    # Showing F1 alongside loss helps illustrate that a
    # decreasing loss does not always mean better class balance performance.

    epochs = range(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="Train")
    plt.plot(epochs, history["val_loss"], label="Validation")
    plt.title(f"Loss - {title}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_f1"], label="Train")
    plt.plot(epochs, history["val_f1"], label="Validation")
    plt.title(f"F1 Score - {title}")
    plt.xlabel("Epoch")
    plt.ylabel("F1 Score")
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, class_names: Sequence[str], output_path: Path
) -> None:
    """Save a confusion matrix heatmap."""

    # We annotate counts to help build intuition about how
    # metrics derive from TP/FP/TN/FN.

    matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(4, 4))
    plt.imshow(matrix, interpolation="nearest", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    thresh = matrix.max() / 2.0 if matrix.size else 0.5
    for i, j in np.ndindex(matrix.shape):
        plt.text(
            j,
            i,
            format(matrix[i, j], "d"),
            horizontalalignment="center",
            color="white" if matrix[i, j] > thresh else "black",
        )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_roc_curves(
    baselines: Dict[str, Tuple[np.ndarray, np.ndarray]],
    output_path: Path,
) -> None:
    """Plot ROC curves for multiple models."""

    # ROC curves compare ranking quality across the full
    # threshold range; AUC is a threshold-independent summary.

    plt.figure(figsize=(6, 6))
    for label, (y_true, y_prob) in baselines.items():
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{label} (AUC={roc_auc:.3f})")

    plt.plot([0, 1], [0, 1], "k--", label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend(loc="lower right")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_pr_curves(
    baselines: Dict[str, Tuple[np.ndarray, np.ndarray]],
    output_path: Path,
) -> None:
    """Plot Precision-Recall curves and AP for multiple models."""

    # PR curves are often preferable under class imbalance
    # and when recall is a priority, common in medical detection tasks.

    plt.figure(figsize=(6, 6))
    for label, (y_true, y_prob) in baselines.items():
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
        plt.plot(recall, precision, label=f"{label} (AP={ap:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves")
    plt.legend(loc="lower left")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def compute_binary_confusion_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, pos_index: int
) -> Dict[str, float]:
    """Compute TP, FP, TN, FN and derived rates for binary classification."""
    # We compute both rates (TPR/FPR) and predictive values
    # (precision/NPV) to illustrate multiple facets of performance.
    y_true_bin = (y_true == pos_index).astype(int)
    y_pred_bin = (y_pred == pos_index).astype(int)
    cmat = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1])
    if cmat.shape == (2, 2):
        tn, fp, fn, tp = cmat.ravel()
    else:
        tn = fp = fn = tp = 0

    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    acc = (tp + tn) / max(1, (tp + tn + fp + fn))

    return {
        "TP": float(tp),
        "FP": float(fp),
        "TN": float(tn),
        "FN": float(fn),
        "TPR": float(tpr),
        "TNR": float(tnr),
        "FPR": float(fpr),
        "FNR": float(fnr),
        "precision": float(precision),
        "recall": float(tpr),
        "npv": float(npv),
        "accuracy": float(acc),
    }


def plot_metrics_bars(
    metrics_map: Dict[str, Dict[str, float]], output_path: Path, keys: Sequence[str]
) -> None:
    """Bar chart comparing metrics across variants."""
    # Grouped bars provide a quick side-by-side comparison of
    # operating points and model families.
    labels = list(metrics_map.keys())
    x = np.arange(len(labels))
    width = 0.12

    plt.figure(figsize=(max(7, len(labels) * 1.6), 4))
    for idx, key in enumerate(keys):
        values = [metrics_map[lbl].get(key, 0.0) for lbl in labels]
        plt.bar(x + idx * width, values, width=width, label=key)
    plt.xticks(x + (len(keys) - 1) * width / 2, labels, rotation=15)
    plt.ylabel("Score")
    plt.title("Metric Comparison")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


@dataclasses.dataclass
class TrainingConfig:
    strong_data_dir: Path
    weak_data_dir: Path
    batch_size: int = 16
    val_split: float = 0.2
    test_split: float = 0.2
    seed: int = 42
    image_size: int = 224
    num_workers: int = 2
    # Device selection: "auto" picks CUDA if available, else CPU
    device: str = "auto"
    # Thresholding to bias toward high recall for the positive class
    positive_class: str = "cancer"
    target_recall: Optional[float] = None
    # Optional operating constraints to avoid excessive false positives
    min_precision: Optional[float] = None
    max_fpr: Optional[float] = None
    f_beta: float = 2.0
    baseline_epochs: int = 10
    weak_pretrain_epochs: int = 5
    finetune_epochs: int = 8
    pseudo_label_threshold: float = 0.7
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    early_stopping_patience: int = 3
    output_dir: Path = Path("outputs")
    results_table: Path = Path("outputs/tables/results_comparison.csv")
    baseline_curve_path: Path = Path("outputs/figures/train_curves_baseline.png")
    semi_curve_path: Path = Path("outputs/figures/train_curves_semi.png")
    baseline_confusion_path: Path = Path(
        "outputs/figures/confusion_matrix_baseline.png"
    )
    semi_confusion_path: Path = Path(
        "outputs/figures/confusion_matrix_semi.png"
    )
    roc_curve_path: Path = Path("outputs/figures/roc_curves.png")
    history_path: Path = Path("outputs/notes/training_history.json")
    baseline_checkpoint: Path = Path("outputs/models/baseline_resnet18.pt")
    semi_checkpoint: Path = Path("outputs/models/semi_resnet18.pt")
    # Optional: filter unlabeled pool to a cohort CSV (paths column)
    unlabeled_cohort_csv: Optional[Path] = None
    # Artifacts for deployment/ops
    operating_point_path: Path = Path("outputs/notes/operating_point.json")
    triage_csv_path: Path = Path("outputs/tables/unlabeled_predictions_semi.csv")


def run_pipeline(config: TrainingConfig) -> Dict[str, Dict[str, float]]:
    """Execute the full training and evaluation workflow."""

    # The pipeline has 3 phases:
    # 1) Train a supervised baseline on strong labels.
    # 2) Use it to pseudo-label the weak pool and pretrain a second model.
    # 3) Fine-tune that model on the strong labels again, then evaluate both.
    # Each phase writes artifacts so students can inspect and compare.

    set_seed(config.seed)
    # Resolve device preference
    if config.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(config.device)
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
    # Resolve positive class index for binary metrics/thresholding
    if config.positive_class not in base_dataset.class_to_idx:
        raise ValueError(
            f"Positive class '{config.positive_class}' not found in dataset classes: {base_dataset.classes}"
        )
    pos_index = int(base_dataset.class_to_idx[config.positive_class])
    criterion = nn.CrossEntropyLoss()

    # -------------------------- Baseline training --------------------------
    # Why start with a baseline? It sets an honest reference point and
    # produces the teacher needed for pseudo-labelling the weak pool.
    baseline_model = create_model(num_classes=num_classes, pretrained=True).to(device)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, baseline_model.parameters()),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    # AdamW: decoupled weight decay tends to provide stable generalization
    # on vision backbones. The defaults here are conservative and appropriate
    # for fine-tuning; increase LR cautiously if underfitting.
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=2, factor=0.5
    )
    # ReduceLROnPlateau + early stopping: LR is reduced when validation loss
    # stalls (plateau), giving the optimizer a chance to make progress; early
    # stopping then terminates if no improvement is seen for several steps.
    # This pairing avoids overfitting and reduces wasted epochs.

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

    # Evaluate baseline on test (argmax) always
    base_arg_metrics, base_arg_y_true, base_arg_y_pred, base_y_prob, _ = (
        evaluate_model(baseline_model, test_loader, device)
    )

    # Threshold tuning (optional): choose threshold on validation to reach target recall
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
        # No threshold tuning: mirror argmax metrics for the thresholded slot for downstream tables/plots
        thr_baseline = None
        base_thr_metrics = dict(base_arg_metrics)
        base_thr_y_true, base_thr_y_pred, base_thr_y_prob = base_arg_y_true, base_arg_y_pred, base_y_prob
        base_thr_metrics["threshold"] = None
        base_thr_metrics["target_recall"] = None
        base_thr_metrics["min_precision"] = None
        base_thr_metrics["max_fpr"] = None
        base_thr_metrics["threshold_policy"] = "disabled"

    base_thr_metrics["training_time_sec"] = baseline_time

    plot_training_curves(baseline_history, config.baseline_curve_path, "Baseline")

    # ------------------- Semi-supervised pseudo labelling -------------------
    # We generate pseudo-labels from the baseline to turn unlabeled images
    # into a (noisy) supervised dataset. We freeze most of the new model
    # at first to reduce overfitting to noise, then unfreeze for fine-tuning.
    unlabeled_dataset = UnlabeledImageDataset(
        config.weak_data_dir, transform=transforms_map["eval"]
    )
    # Optionally filter the unlabeled pool using a cohort CSV (from clustering)
    if config.unlabeled_cohort_csv is not None:
        cohort_path = Path(config.unlabeled_cohort_csv)
        if not cohort_path.exists():
            raise FileNotFoundError(f"Cohort CSV not found: {cohort_path}")
        cohort_df = pd.read_csv(cohort_path)
        if "path" not in cohort_df.columns:
            raise ValueError("Cohort CSV must contain a 'path' column")
        # Normalize cohort paths to absolute filesystem paths. CSV may contain
        # dataset-root-relative paths like 'sans_label/xxx.jpg' or file names.
        # We accept the following candidates per entry:
        #  - weak_data_dir / pp
        #  - weak_data_dir / (pp without the leading weak folder name)
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
                    # bare filename
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
            raise RuntimeError("Cohort filtering removed all unlabeled images; check the CSV paths match --weak-data-dir.")
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
        filter(lambda p: p.requires_grad, semi_model.parameters()),
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

    # Fine-tuning stage
    # After learning a head from pseudo-labels, we unfreeze the backbone and
    # adapt all weights to the (clean) strong labels, usually with a lower LR.
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

    # Performance note: Consider enabling AMP (mixed precision) and channels-last
    # memory format on modern GPUs to increase throughput. This repository keeps
    # the reference path deterministic/simpler, but production workloads often
    # benefit from these optimizations if numerical stability is acceptable.

    # Semi-supervised evaluation: always compute argmax
    semi_arg_metrics, semi_arg_y_true, semi_arg_y_pred, semi_y_prob, _ = evaluate_model(
        semi_model, test_loader, device
    )

    # Optional threshold tuning for semi model
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
        semi_thr_y_true, semi_thr_y_pred, semi_thr_y_prob = semi_arg_y_true, semi_arg_y_pred, semi_y_prob
        semi_thr_metrics["threshold"] = None
        semi_thr_metrics["target_recall"] = None
        semi_thr_metrics["min_precision"] = None
        semi_thr_metrics["max_fpr"] = None
        semi_thr_metrics["threshold_policy"] = "disabled"

    semi_thr_metrics["training_time_sec"] = semi_time

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

    # Confusion matrices for argmax and thresholded variants
    plot_confusion_matrix(base_arg_y_true, base_arg_y_pred, base_dataset.classes, config.baseline_confusion_path)
    plot_confusion_matrix(base_thr_y_true, base_thr_y_pred, base_dataset.classes, Path("outputs/figures/confusion_matrix_baseline_thresholded.png"))
    plot_confusion_matrix(semi_arg_y_true, semi_arg_y_pred, base_dataset.classes, config.semi_confusion_path)
    plot_confusion_matrix(semi_thr_y_true, semi_thr_y_pred, base_dataset.classes, Path("outputs/figures/confusion_matrix_semi_thresholded.png"))

    # ROC expects binary labels for the positive class
    baseline_y_true_bin = (base_thr_y_true == pos_index).astype(int)
    semi_y_true_bin = (semi_thr_y_true == pos_index).astype(int)
    plot_roc_curves(
        {
            "Baseline": (baseline_y_true_bin, base_thr_y_prob),
            "Semi-supervised": (semi_y_true_bin, semi_thr_y_prob),
        },
        config.roc_curve_path,
    )

    # Precision-Recall curves
    plot_pr_curves(
        {
            "Baseline": (baseline_y_true_bin, base_thr_y_prob),
            "Semi-supervised": (semi_y_true_bin, semi_thr_y_prob),
        },
        Path("outputs/figures/pr_curves.png"),
    )

    # Detailed confusion-derived metrics for all variants
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

    # Metrics bar comparison
    plot_metrics_bars(
        detailed_rows,
        Path("outputs/figures/metrics_comparison.png"),
        keys=["TPR", "FPR", "TNR", "precision", "accuracy"],
    )

    # Keep summary focusing on thresholded decisions
    results_df = pd.DataFrame.from_dict(
        {
            "baseline_thresholded": base_thr_metrics,
            "semi_thresholded": semi_thr_metrics,
        },
        orient="index",
    )
    config.results_table.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(config.results_table)

    # ------------------- Operating point manifest -------------------
    try:
        op_payload = {
            "model": "semi_supervised_resnet18",
            "checkpoint": str(config.semi_checkpoint),
            "positive_class": config.positive_class,
            "threshold": (None if 'threshold' not in semi_thr_metrics else semi_thr_metrics.get("threshold")),
            "policy": semi_thr_metrics.get("threshold_policy"),
            "target_recall": config.target_recall,
            "min_precision": config.min_precision,
            "max_fpr": config.max_fpr,
            "seed": config.seed,
            "created_at": datetime.now().isoformat(),
        }
        config.operating_point_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config.operating_point_path, "w", encoding="utf-8") as fp:
            json.dump(op_payload, fp, indent=2)
    except Exception as exc:
        LOGGER.warning("Failed to write operating_point.json: %s", exc)

    # ------------------- Triage CSV for unlabeled pool -------------------
    try:
        triage_rows: List[Dict[str, Any]] = []
        triage_thr = semi_thr_metrics.get("threshold")
        if triage_thr is not None:
            # Reuse the unlabeled dataset/loader (filtered if cohort CSV was provided)
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


def parse_args(args: Optional[Sequence[str]] = None) -> TrainingConfig:
    """Parse command-line arguments into a :class:`TrainingConfig`."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strong-data-dir",
        type=Path,
        required=True,
        help="Path to the strongly labelled dataset (ImageFolder layout).",
    )
    parser.add_argument(
        "--weak-data-dir",
        type=Path,
        required=True,
        help="Path to the weakly labelled/unlabelled dataset (flat directory).",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--test-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument(
        "--num-workers", type=int, default=2, help="Dataloader worker processes"
    )
    parser.add_argument("--baseline-epochs", type=int, default=10)
    parser.add_argument("--weak-pretrain-epochs", type=int, default=5)
    parser.add_argument("--finetune-epochs", type=int, default=8)
    parser.add_argument("--pseudo-threshold", type=float, default=0.7)
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
        help="Optional target recall for validation-based threshold selection (0-1). If omitted, no threshold tuning is applied.",
    )
    parser.add_argument(
        "--min-precision",
        type=float,
        default=None,
        help="Optional minimum precision constraint for threshold selection (0-1)",
    )
    parser.add_argument(
        "--max-fpr",
        type=float,
        default=None,
        help="Optional maximum false positive rate constraint for threshold selection (0-1)",
    )
    parser.add_argument(
        "--f-beta",
        type=float,
        default=2.0,
        help="F-beta fallback when constraints cannot be met; beta>1 favors recall",
    )
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
    parser.add_argument(
        "--unlabeled-cohort-csv",
        type=Path,
        default=None,
        help="Optional CSV listing unlabeled image paths to include in pseudo-labeling (column: path). Useful to restrict to DBSCAN non-noise cohorts.",
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
        weak_pretrain_epochs=parsed.weak_pretrain_epochs,
        finetune_epochs=parsed.finetune_epochs,
        pseudo_label_threshold=parsed.pseudo_threshold,
        learning_rate=parsed.learning_rate,
        weight_decay=parsed.weight_decay,
        early_stopping_patience=parsed.early_stopping,
    device=parsed.device,
        output_dir=parsed.output_dir,
        results_table=parsed.output_dir / "tables/results_comparison.csv",
        baseline_curve_path=parsed.output_dir / "figures/train_curves_baseline.png",
        semi_curve_path=parsed.output_dir / "figures/train_curves_semi.png",
        baseline_confusion_path=
        parsed.output_dir / "figures/confusion_matrix_baseline.png",
        semi_confusion_path=parsed.output_dir / "figures/confusion_matrix_semi.png",
        roc_curve_path=parsed.output_dir / "figures/roc_curves.png",
        history_path=parsed.output_dir / "notes/training_history.json",
        baseline_checkpoint=parsed.output_dir / "models/baseline_resnet18.pt",
        semi_checkpoint=parsed.output_dir / "models/semi_resnet18.pt",
        unlabeled_cohort_csv=parsed.unlabeled_cohort_csv,
    )


def main(args: Optional[Sequence[str]] = None) -> None:
    """Entry-point wrapper used by ``python -m``."""

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s:%(name)s:%(message)s",
    )

    config = parse_args(args)
    metrics = run_pipeline(config)
    LOGGER.info("Experiment complete. Metrics:\n%s", json.dumps(metrics, indent=2))


if __name__ == "__main__":  # pragma: no cover - CLI execution
    main()

