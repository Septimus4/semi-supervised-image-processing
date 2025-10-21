"""CLI wrapper for semi-supervised training.

This script now delegates the heavy lifting to ``training.semi_supervised``
and shares utilities from ``training.common``. It preserves the same CLI
as before for backward compatibility.

Usage:
    python -m src.semi_supervised_training --strong-data-dir <path> --weak-data-dir <path>
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional, Sequence

from training.common import TrainingConfig
from training.semi_supervised import run_pipeline

LOGGER = logging.getLogger(__name__)

LEGACY_IGNORE = '''
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

'''


## Heavy-lifting lives in training.semi_supervised.run_pipeline


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
        baseline_confusion_path= parsed.output_dir / "figures/confusion_matrix_baseline.png",
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

