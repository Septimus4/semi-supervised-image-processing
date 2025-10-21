from __future__ import annotations

import argparse
import json
from typing import Sequence
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets

from src.training.common import (
    build_transforms,
    TransformSubset,
    create_model,
)


def compute_probs(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    pos_index: int,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    y_true: List[int] = []
    y_prob: List[float] = []
    with torch.no_grad():
        for batch in loader:
            inputs, labels = batch[:2]
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)[:, pos_index]
            y_true.extend(labels.cpu().numpy().tolist())
            y_prob.extend(probs.cpu().numpy().tolist())
    return np.array(y_true), np.array(y_prob)


def confusion_from_threshold(
    y_true: np.ndarray, y_prob: np.ndarray, pos_index: int, thr: float
) -> Dict[str, float]:
    # We convert to a binary task w.r.t. the positive class.
    # Thresholding the positive-class probability allows us to trade off
    # recall vs. precision. We return confusion-derived metrics for teaching.
    y_true_bin = (y_true == pos_index).astype(int)
    y_pred_bin = (y_prob >= thr).astype(int)
    tp = float(((y_true_bin == 1) & (y_pred_bin == 1)).sum())
    tn = float(((y_true_bin == 0) & (y_pred_bin == 0)).sum())
    fp = float(((y_true_bin == 0) & (y_pred_bin == 1)).sum())
    fn = float(((y_true_bin == 1) & (y_pred_bin == 0)).sum())

    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    acc = (tp + tn) / max(1.0, (tp + tn + fp + fn))
    return {
        "threshold": float(thr),
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "TPR": float(tpr),
        "TNR": float(tnr),
        "FPR": float(fpr),
        "precision": float(precision),
        "accuracy": float(acc),
    }


def load_splits(history_path: Path) -> Dict[str, np.ndarray]:
    data = json.loads(history_path.read_text())
    return {k: np.array(v) for k, v in data["splits"].items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Threshold sweep on test split")
    parser.add_argument("--strong-data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--model",
        type=str,
        choices=["baseline", "semi"],
        default="semi",
        help="Which trained checkpoint to evaluate",
    )
    parser.add_argument("--positive-class", type=str, default="cancer")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    # Use the requested device when available; gracefully
    # fall back to CPU so the tool can be run on any machine in class.
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    base_dataset = datasets.ImageFolder(str(args.strong_data_dir), transform=None)
    if args.positive_class not in base_dataset.class_to_idx:
        raise SystemExit(
            f"Positive class '{args.positive_class}' not found in {base_dataset.classes}"
        )
    pos_index = int(base_dataset.class_to_idx[args.positive_class])

    # Load splits from training
    history_path = args.output_dir / "notes/training_history.json"
    splits = load_splits(history_path)
    test_idx = splits["test"]

    transforms_map = build_transforms(224)
    # Ensure indices are a plain Python sequence for type checkers and Dataset API
    test_indices: Sequence[int] = list(map(int, list(test_idx)))
    test_dataset = TransformSubset(
        base_dataset, test_indices, transform=transforms_map["eval"], return_paths=False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    # Load checkpoint
    num_classes = len(base_dataset.classes)
    model = create_model(num_classes=num_classes, pretrained=True).to(device)
    ckpt_path = (
        args.output_dir / "models" / ("baseline_resnet18.pt" if args.model == "baseline" else "semi_resnet18.pt")
    )
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)

    y_true, y_prob = compute_probs(model, test_loader, device, pos_index)

    # Sweep thresholds from 1.0 → 0.0 so we can pick the
    # largest threshold that achieves a desired recall (fewer false positives).
    thresholds = np.unique(np.concatenate(([0.0], y_prob, [1.0])))[::-1]
    rows = [confusion_from_threshold(y_true, y_prob, pos_index, thr) for thr in thresholds]

    out_dir = args.output_dir / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"threshold_sweep_{args.model}.csv"
    import pandas as pd

    pd.DataFrame(rows).to_csv(out_csv, index=False)

    # Report best threshold for TPR==1.0 with minimal FPR (pick largest threshold)
    # In screening scenarios, missing a positive can be worse
    # than a false alarm. We therefore prioritise TPR≈1.0 and then pick the
    # highest possible threshold to keep FPR in check.
    sweep = pd.DataFrame(rows)
    tpr1 = sweep[sweep["TPR"] >= 0.999999]
    if not tpr1.empty:
        # highest threshold among those achieving TPR ~ 1
        best = tpr1.sort_values(["threshold"], ascending=False).iloc[0]
        print(json.dumps({
            "best_threshold": float(best["threshold"]),
            "TP": float(best["TP"]),
            "FP": float(best["FP"]),
            "TN": float(best["TN"]),
            "FN": float(best["FN"]),
            "TPR": float(best["TPR"]),
            "FPR": float(best["FPR"]),
            "precision": float(best["precision"]),
            "accuracy": float(best["accuracy"]),
            "csv": str(out_csv),
        }))
    else:
        print(json.dumps({"message": "No threshold achieves TPR=1.0 on test", "csv": str(out_csv)}))


if __name__ == "__main__":
    main()
