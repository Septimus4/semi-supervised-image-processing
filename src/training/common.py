"""Common training utilities for supervised and semi-supervised pipelines.

This module centralizes shared code: reproducibility, transforms, datasets,
loaders, model creation, generic training/evaluation loops, plotting, and
thresholding helpers. Both supervised and semi-supervised pipelines import
from here to avoid duplication.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import logging
import math
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
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

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class TrainingConfig:
    strong_data_dir: Path
    weak_data_dir: Path
    batch_size: int = 16
    val_split: float = 0.2
    test_split: float = 0.2
    seed: int = 42
    image_size: int = 224
    num_workers: int = 2
    device: str = "auto"  # "auto" | "cpu" | "cuda"
    positive_class: str = "cancer"
    target_recall: Optional[float] = None
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
    baseline_confusion_path: Path = Path("outputs/figures/confusion_matrix_baseline.png")
    semi_confusion_path: Path = Path("outputs/figures/confusion_matrix_semi.png")
    roc_curve_path: Path = Path("outputs/figures/roc_curves.png")
    history_path: Path = Path("outputs/notes/training_history.json")
    baseline_checkpoint: Path = Path("outputs/models/baseline_resnet18.pt")
    semi_checkpoint: Path = Path("outputs/models/semi_resnet18.pt")
    unlabeled_cohort_csv: Optional[Path] = None
    operating_point_path: Path = Path("outputs/notes/operating_point.json")
    triage_csv_path: Path = Path("outputs/tables/unlabeled_predictions_semi.csv")


# ---------------------------------------------------------------------------
# Reproducibility and transforms
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_transforms(image_size: int = 224) -> Dict[str, transforms.Compose]:
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


# ---------------------------------------------------------------------------
# Datasets and loaders
# ---------------------------------------------------------------------------

class TransformSubset(Dataset):
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

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        image, label = self.dataset[self.indices[idx]]
        if self.transform is not None:
            image = self.transform(image)
        if self.return_paths:
            path = self.dataset.samples[self.indices[idx]][0]
            return image, label, path
        return image, label


class UnlabeledImageDataset(Dataset):
    def __init__(self, root_dir: Path, transform: Optional[transforms.Compose] = None) -> None:
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

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        path = self.image_paths[idx]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, str(path)


class PseudoLabeledDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[Tuple[str, int]],
        transform: Optional[transforms.Compose] = None,
    ) -> None:
        self.samples = list(samples)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
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
    label_array = np.array(labels)
    class_sample_count = np.bincount(label_array)
    if len(np.nonzero(class_sample_count)[0]) < 2:
        LOGGER.warning(
            "Only one class present in labels; using uniform sampling instead of balancing."
        )
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
    base_dataset = datasets.ImageFolder(strong_data_dir, transform=None)
    targets = np.array(base_dataset.targets)
    train_idx, val_idx, test_idx = stratified_split(targets.tolist(), val_split, test_split, seed)
    split_indices = {"train": train_idx, "val": val_idx, "test": test_idx}

    train_dataset = TransformSubset(base_dataset, list(train_idx), transform=transforms_map["train"])
    val_dataset = TransformSubset(base_dataset, list(val_idx), transform=transforms_map["eval"], return_paths=True)
    test_dataset = TransformSubset(base_dataset, list(test_idx), transform=transforms_map["eval"], return_paths=True)

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


# ---------------------------------------------------------------------------
# Model and training loop
# ---------------------------------------------------------------------------

def create_model(num_classes: int, pretrained: bool = True) -> nn.Module:
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def compute_accuracy_f1(y_true: Sequence[int], y_pred: Sequence[int]) -> Tuple[float, float]:
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
    history: Dict[str, List[float]] = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "train_f1": [],
        "val_f1": [],
    }

    best_state = model.state_dict()
    best_val_loss = math.inf
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        train_losses: List[float] = []
        y_true_train: List[int] = []
        y_pred_train: List[int] = []

        for inputs, labels in train_loader:
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
        val_loss, val_acc, val_f1 = evaluate_on_loader(model, val_loader, criterion, device)

        if scheduler is not None:
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
            best_val_loss = val_loss
            best_state = model.state_dict()
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


# ---------------------------------------------------------------------------
# Evaluation helpers and plots
# ---------------------------------------------------------------------------

def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    pos_index: Optional[int] = None,
    threshold: Optional[float] = None,
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray, np.ndarray, List[str]]:
    model.eval()
    y_true: List[int] = []
    y_pred: List[int] = []
    y_prob: List[float] = []
    sample_paths: List[str] = []

    with torch.no_grad():
        for batch in data_loader:
            inputs, labels = batch[:2]
            extras = batch[2:] if len(batch) > 2 else []
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


def plot_training_curves(history: Dict[str, List[float]], output_path: Path, title: str) -> None:
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


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, class_names: Sequence[str], output_path: Path) -> None:
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


def plot_roc_curves(baselines: Dict[str, Tuple[np.ndarray, np.ndarray]], output_path: Path) -> None:
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


def plot_pr_curves(baselines: Dict[str, Tuple[np.ndarray, np.ndarray]], output_path: Path) -> None:
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


def compute_binary_confusion_metrics(y_true: np.ndarray, y_pred: np.ndarray, pos_index: int) -> Dict[str, float]:
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
        "accuracy": float(acc),
    }


def plot_metrics_bars(metrics_map: Dict[str, Dict[str, float]], output_path: Path, keys: Sequence[str]) -> None:
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


# ---------------------------------------------------------------------------
# Threshold selection helpers (shared)
# ---------------------------------------------------------------------------

def find_threshold_for_target_recall(y_true_bin: np.ndarray, y_prob: np.ndarray, target_recall: float) -> float:
    if y_true_bin.sum() == 0:
        return 0.5
    thresholds = np.unique(np.concatenate(([0.0], y_prob)))
    thresholds.sort()
    best_thr = thresholds[0]
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
        if precision + recall > 0:
            beta2 = f_beta * f_beta
            fbeta = (1 + beta2) * (precision * recall) / (beta2 * precision + recall)
        else:
            fbeta = 0.0
        return recall, precision, fpr, tp, fp, fbeta

    feasible: List[Tuple[float, float, float, float]] = []
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
        thr, recall, precision, fpr = sorted(feasible, key=lambda x: x[0])[-1]
        return float(thr), {
            "policy": "constrained",
            "recall": float(recall),
            "precision": float(precision),
            "fpr": float(fpr),
        }

    scored: List[Tuple[float, float, float, float]] = []
    for thr in thresholds:
        recall, precision, fpr, _tp, _fp, fbeta = stats_at(thr)
        scored.append((fbeta, float(thr), recall, precision))
    scored.sort(key=lambda x: (x[0], x[1]))
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
