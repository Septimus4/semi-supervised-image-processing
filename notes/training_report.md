# Task 4 – Semi-Supervised vs. Supervised Training Report (updated)

## Overview
This report compares a supervised baseline to a semi-supervised pipeline that pre-trains on pseudo-labelled images from the weak pool before fine-tuning on the strongly labelled cohort. Executed via `python -m src.semi_supervised_training`; implementation uses shared utilities in `src/training/common.py` and the pipeline in `src/training/semi_supervised.py`.

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
* Threshold tuning: optional. Earlier runs used validation-based tuning with `--target-recall 1.0` for the positive class. Current cohort-guided run used `--target-recall 0.98` with `--min-precision 0.60` (constrained policy).

## Results – Cohort-guided run (DBSCAN non-noise)
Values sourced from `outputs/tables/results_comparison.csv` (generated after training with `--unlabeled-cohort-csv outputs/tables/unlabeled_cohort_dbscan.csv`).

Unlabeled cohort: 1,405 images in CSV; pseudo-labeled used for pre-train: 1,275 (threshold ≥ 0.70).

| Model | Accuracy | Precision | Recall | F1 | Training time (s) | Threshold | Target recall |
| --- | --- | --- | --- | --- | ---:| ---:| ---:|
| Baseline (thresholded) | 0.90 | 0.90 | 0.90 | 0.90 | 2.49 | 0.309 | 0.98 |
| Semi-supervised (thresholded) | 0.95 | 1.00 | 0.90 | 0.947 | 4.70 | 0.879 | 0.98 |

For full details including argmax metrics, confusion matrices, and PR/ROC curves, see:
* Tables: `outputs/tables/{results_comparison.csv, results_comparison_detailed.csv}`
* Curves: `outputs/figures/{train_curves_baseline.png, train_curves_semi.png, roc_curves.png, pr_curves.png}`
* Confusion matrices: `outputs/figures/confusion_matrix_*`

## Visual diagnostics
* **Training dynamics** – `outputs/figures/train_curves_baseline.png`, `outputs/figures/train_curves_semi.png`.
* **Confusion matrices** – argmax and thresholded variants for both models under `outputs/figures/`.
* **ROC/PR curves** – `outputs/figures/roc_curves.png`, `outputs/figures/pr_curves.png`.

## Interpretation & next steps
* With cohort-guided pseudo-labeling, the semi-supervised model improved to accuracy 0.95, precision 1.00, recall 0.90 at a higher decision threshold; baseline remained at 0.90/0.90.
* Cohort selection (DBSCAN non-noise) improves pseudo-label quality while keeping coverage high.
* If recall must be ≥1.0 on validation, keep threshold tuning enabled; otherwise, omit `--target-recall` to use argmax decisions.
* Potential improvements: tune pseudo-label threshold, iterate pseudo-labelling, or add mixup/cutmix during pseudo pre-training.
