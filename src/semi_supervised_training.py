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
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

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
    precision_recall_fscore_support,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import datasets, models, transforms
from PIL import Image


LOGGER = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Ensure reproducible behaviour across libraries."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_transforms(image_size: int = 224) -> Dict[str, transforms.Compose]:
    """Construct train and evaluation transforms."""

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

    label_array = np.array(labels)
    class_sample_count = np.bincount(label_array)
    if len(np.nonzero(class_sample_count)[0]) < 2:
        LOGGER.warning(
            "Only one class present in labels; using uniform sampling instead of balancing."
        )
        return WeightedRandomSampler(
            weights=torch.ones(len(label_array), dtype=torch.double),
            num_samples=len(label_array),
            replacement=True,
        )

    weight_per_class = 1.0 / class_sample_count
    samples_weight = weight_per_class[label_array]
    return WeightedRandomSampler(
        weights=torch.from_numpy(samples_weight).double(),
        num_samples=len(samples_weight),
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

    base_dataset = datasets.ImageFolder(strong_data_dir, transform=None)
    targets = np.array(base_dataset.targets)
    train_idx, val_idx, test_idx = stratified_split(targets, val_split, test_split, seed)

    split_indices = {"train": train_idx, "val": val_idx, "test": test_idx}

    train_dataset = TransformSubset(
        base_dataset, train_idx, transform=transforms_map["train"]
    )
    val_dataset = TransformSubset(
        base_dataset, val_idx, transform=transforms_map["eval"], return_paths=True
    )
    test_dataset = TransformSubset(
        base_dataset, test_idx, transform=transforms_map["eval"], return_paths=True
    )

    train_targets = targets[train_idx]
    sampler = make_balanced_sampler(train_targets)

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
    scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
    num_epochs: int = 10,
    early_stopping_patience: int = 3,
    model_path: Optional[Path] = None,
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    """Generic training loop with early stopping."""

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
            scheduler.step(val_loss)

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
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Evaluate the model and capture predictions for reporting."""

    model.eval()
    y_true: List[int] = []
    y_pred: List[int] = []
    y_prob: List[float] = []
    sample_paths: List[str] = []

    with torch.no_grad():
        for batch in data_loader:
            inputs, labels = batch[:2]
            extras = batch[2:] if len(batch) > 2 else []
            paths = extras[0] if extras else [None] * len(labels)

            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            probabilities = torch.softmax(outputs, dim=1)[:, 1]
            predictions = outputs.argmax(dim=1)

            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(predictions.cpu().numpy().tolist())
            y_prob.extend(probabilities.cpu().numpy().tolist())
            sample_paths.extend(list(paths))

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


def generate_pseudo_labels(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    threshold: float = 0.7,
) -> List[Tuple[str, int, float]]:
    """Generate pseudo labels for weakly labelled data."""

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

    matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(4, 4))
    plt.imshow(matrix, interpolation="nearest", cmap=plt.cm.Blues)
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


def run_pipeline(config: TrainingConfig) -> Dict[str, Dict[str, float]]:
    """Execute the full training and evaluation workflow."""

    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    criterion = nn.CrossEntropyLoss()

    # -------------------------- Baseline training --------------------------
    baseline_model = create_model(num_classes=num_classes, pretrained=True).to(device)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, baseline_model.parameters()),
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

    baseline_metrics, baseline_y_true, baseline_y_pred, baseline_y_prob, _ = (
        evaluate_model(baseline_model, test_loader, device)
    )
    baseline_metrics["training_time_sec"] = baseline_time

    plot_training_curves(baseline_history, config.baseline_curve_path, "Baseline")
    plot_confusion_matrix(
        baseline_y_true,
        baseline_y_pred,
        base_dataset.classes,
        config.baseline_confusion_path,
    )

    # ------------------- Semi-supervised pseudo labelling -------------------
    unlabeled_dataset = UnlabeledImageDataset(
        config.weak_data_dir, transform=transforms_map["eval"]
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

    semi_metrics, semi_y_true, semi_y_pred, semi_y_prob, _ = evaluate_model(
        semi_model, test_loader, device
    )
    semi_metrics["training_time_sec"] = semi_time

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

    plot_confusion_matrix(
        semi_y_true,
        semi_y_pred,
        base_dataset.classes,
        config.semi_confusion_path,
    )

    plot_roc_curves(
        {
            "Baseline": (baseline_y_true, baseline_y_prob),
            "Semi-supervised": (semi_y_true, semi_y_prob),
        },
        config.roc_curve_path,
    )

    results_df = pd.DataFrame.from_dict(
        {
            "baseline_supervised": baseline_metrics,
            "semi_supervised": semi_metrics,
        },
        orient="index",
    )
    config.results_table.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(config.results_table)

    return {
        "baseline_supervised": baseline_metrics,
        "semi_supervised": semi_metrics,
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
    parser.add_argument("--baseline-epochs", type=int, default=10)
    parser.add_argument("--weak-pretrain-epochs", type=int, default=5)
    parser.add_argument("--finetune-epochs", type=int, default=8)
    parser.add_argument("--pseudo-threshold", type=float, default=0.7)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--early-stopping", type=int, default=3)
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
        baseline_epochs=parsed.baseline_epochs,
        weak_pretrain_epochs=parsed.weak_pretrain_epochs,
        finetune_epochs=parsed.finetune_epochs,
        pseudo_label_threshold=parsed.pseudo_threshold,
        learning_rate=parsed.learning_rate,
        weight_decay=parsed.weight_decay,
        early_stopping_patience=parsed.early_stopping,
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

