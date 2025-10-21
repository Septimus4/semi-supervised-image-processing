# Task 4 – Semi-Supervised vs. Supervised Training Report (updated)

## Overview
This report compares a supervised baseline to a semi-supervised pipeline that pre-trains on pseudo-labelled images from the weak pool before fine-tuning on the strongly labelled cohort. Implemented in `src/semi_supervised_training.py` and executed via `python -m src.semi_supervised_training`.

## Data preparation
* **Strong labels** – `mri_dataset_brain_cancer_oc/avec_labels` (classes: `cancer`, `normal`).
* **Weak pool** – `mri_dataset_brain_cancer_oc/sans_label` used for pseudo-labelling.
* Stratified splits (seed=42) are persisted in `outputs/notes/training_history.json` for reproducibility.

## Training configuration (current baseline run)
* Backbone: `torchvision.resnet18` (ImageNet weights).
* Image size: 224×224 with standard normalisation; light augmentation on train.
* Optimiser: AdamW (LR 1e-4, weight decay 1e-4) with ReduceLROnPlateau (patience=2).
* Early stopping: patience=3 on validation loss.
* Batch size / workers: 64 / 8 (CUDA).
* Semi-supervised:
  1. Train baseline on strong labels.
  2. Pseudo-label weak pool at confidence ≥ 0.70 → 1,082 samples.
  3. Pre-train a new model on pseudo-labels with backbone frozen.
  4. Unfreeze and fine-tune on strong train split.
* Threshold tuning: optional. This run used validation-based tuning with `--target-recall 1.0` for the positive class.

## Results (thresholded operating point)
Values sourced from `outputs/tables/results_comparison.csv`.

| Model | Accuracy | Precision | Recall | F1 | Training time (s) | Threshold | Target recall |
| --- | --- | --- | --- | --- | ---:| ---:| ---:|
| Baseline (thresholded) | 0.70 | 0.625 | 1.00 | 0.769 | 1.75 | 0.084 | 1.0 |
| Semi-supervised (thresholded) | 0.90 | 0.90 | 0.90 | 0.90 | 7.30 | 0.138 | 1.0 |

For full details including argmax metrics, confusion matrices, and PR/ROC curves, see:
* Tables: `outputs/tables/{results_comparison.csv, results_comparison_detailed.csv}`
* Curves: `outputs/figures/{train_curves_baseline.png, train_curves_semi.png, roc_curves.png, pr_curves.png}`
* Confusion matrices: `outputs/figures/confusion_matrix_*`

## Visual diagnostics
* **Training dynamics** – `outputs/figures/train_curves_baseline.png`, `outputs/figures/train_curves_semi.png`.
* **Confusion matrices** – argmax and thresholded variants for both models under `outputs/figures/`.
* **ROC/PR curves** – `outputs/figures/roc_curves.png`, `outputs/figures/pr_curves.png`.

## Interpretation & next steps
* The thresholded baseline achieves perfect recall at the cost of lower precision; the semi-supervised model balances precision/recall at 0.90/0.90.
* If recall must be ≥1.0 on validation, keep threshold tuning enabled; otherwise, omit `--target-recall` to use argmax decisions.
* Potential improvements: increase pseudo-label threshold, iterative pseudo-labelling, or add mixup/cutmix during pseudo pre-training.
