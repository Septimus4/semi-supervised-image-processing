# Training

The training script compares a supervised baseline with a semi-supervised variant using pseudo-labeling.

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
