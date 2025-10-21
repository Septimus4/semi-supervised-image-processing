# Training

The training compares a supervised baseline with a semi-supervised variant using pseudo-labeling. You can run either the semi-supervised pipeline (baseline + pseudo-labeling + fine-tune) or a supervised-only baseline.

## Quick start (GPU)
```bash
python -m src.semi_supervised_training \
  --strong-data-dir mri_dataset_brain_cancer_oc/avec_labels \
  --weak-data-dir mri_dataset_brain_cancer_oc/sans_label \
  --baseline-epochs 10 \
  --weak-pretrain-epochs 5 \
  --finetune-epochs 8 \
  --batch-size 128 \
  --image-size 224 \
  --num-workers 16 \
  --device cuda
```

### Supervised-only baseline
```bash
python -m src.supervised_training \
  --strong-data-dir mri_dataset_brain_cancer_oc/avec_labels \
  --baseline-epochs 10 \
  --batch-size 128 \
  --image-size 224 \
  --num-workers 16 \
  --device cuda
```

## What it does
1. Split labeled data into train/val/test with stratification.
2. Train a ResNet-18 baseline (pretrained ImageNet) with a new classification head.
3. Generate pseudo-labels on unlabeled images with the baseline.
4. Pretrain a second model on pseudo-labeled data (freeze backbone), then fine-tune on train.
5. Evaluate both models on the test set and write metrics, curves, and confusion matrices.

## Outputs
- Checkpoints: `outputs/models/{baseline_resnet18.pt, semi_resnet18.pt}`
- Metrics table: `outputs/tables/results_comparison.csv`
- Curves: `outputs/figures/train_curves_*.png`, `roc_curves.png`
- Confusion matrices: `outputs/figures/confusion_matrix_*.png`
- Training history + splits: `outputs/notes/training_history.json`

## Tuning tips
- Increase `--num-workers` based on CPU cores to keep the GPU fed (e.g., 12–20 on a 32-thread CPU).
- Raise `--batch-size` if VRAM allows to improve throughput.
- If validation oscillates, adjust `--learning-rate`, or increase patience via `--early-stopping`.

## Favor near-100% detection (accept some false positives)
To bias toward detecting all cancer cases (high recall), the script can choose a
decision threshold on the validation set to meet a desired recall target for the
positive class, then apply that threshold on the test set.

Use with the semi-supervised pipeline:
```bash
python -m src.semi_supervised_training \
  --strong-data-dir mri_dataset_brain_cancer_oc/avec_labels \
  --weak-data-dir mri_dataset_brain_cancer_oc/sans_label \
  --target-recall 1.0 \
  --positive-class cancer \
  --device cuda
```

The same threshold selection flags are available for the supervised-only CLI:
```bash
python -m src.supervised_training \
  --strong-data-dir mri_dataset_brain_cancer_oc/avec_labels \
  --target-recall 1.0 \
  --positive-class cancer \
  --device cuda
```

Notes:
- The `--positive-class` must match the folder name in `avec_labels/` for the positive class.
- `--target-recall 1.0` aims to catch all positives on validation; if unattainable, the lowest threshold is chosen, maximizing recall.
