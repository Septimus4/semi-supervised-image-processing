# Task 4 – Semi-Supervised vs. Supervised Training Report

## Overview
This document summarises the experiments performed to compare a purely supervised baseline against a semi-supervised pipeline that pre-trains on weakly labelled MRI slices before fine-tuning on the strongly labelled cohort. The workflow is implemented in `src/semi_supervised_training.py` and executed via `python -m src.semi_supervised_training`.

## Data preparation
* **Strong labels** – `mri_dataset_brain_cancer_oc/avec_labels` contains two balanced folders (`cancer`, `normal`).
* **Weak labels** – `mri_dataset_brain_cancer_oc/sans_label` is treated as the weak pool; pseudo-labels are inferred from the baseline model.
* Stratified splitting on the strong dataset (seed=42) produced 60 train / 20 val / 20 test images. Split indices are persisted in `outputs/notes/training_history.json` for reproducibility.

## Training configuration
* Backbone: `torchvision.models.resnet18` (ImageNet weights).
* Image size: 224×224 with standard normalisation and light augmentation (rotation, horizontal flip).
* Optimiser: AdamW (initial LR 3e-4, weight decay 1e-4) with ReduceLROnPlateau.
* Early stopping patience: 2 epochs based on validation loss.
* Semi-supervised strategy:
  1. Train baseline on the strong set.
  2. Generate pseudo-labels on the weak set (confidence ≥ 0.6), yielding 1,370 usable samples.
  3. Pre-train a second model on the pseudo-labelled pool with the backbone frozen.
  4. Unfreeze and fine-tune on the strong training set.

## Results
| Model | Accuracy | Precision | Recall | F1 | Training time (s) |
| --- | --- | --- | --- | --- | --- |
| Supervised baseline | 0.90 | 0.90 | 0.90 | 0.90 | 75.0 |
| Semi-supervised | 0.85 | 0.89 | 0.80 | 0.84 | 196.2 |

*The semi-supervised pipeline did not surpass the baseline on the held-out test set despite the additional data and compute. Pseudo-label quality and class imbalance in the weak pool likely introduced noise that limited gains.*

## Visual diagnostics
* **Training dynamics** – `outputs/figures/train_curves_baseline.png` and `train_curves_semi.png` highlight smoother convergence for the baseline and higher variance during pseudo-label pre-training.
* **Confusion matrices** – `outputs/figures/confusion_matrix_baseline.png` (clear diagonal) vs. `confusion_matrix_semi.png` (extra false negatives).
* **ROC curves** – `outputs/figures/roc_curves.png` plots both models (baseline AUC ≈ 0.95 vs. semi-supervised ≈ 0.92).

## Interpretation & next steps
* The baseline already generalises well; pseudo-labels with a 0.6 threshold introduced enough noise to hurt recall.
* Raising the confidence threshold or iterative re-labelling could improve weak supervision quality.
* Exploring mixup/cutmix during pseudo-label training and monitoring class balance in the sampled weak data are promising future adjustments.
