# Semi-Supervised Image Processing

Utilities for auditing and preprocessing the MRI brain cancer dataset used in
semi-supervised experiments.

## Environment setup

```bash
pip install -e .
```

This installs PyTorch, torchvision, and the supporting scientific Python stack
needed by the audit and feature extraction scripts.

## Dataset layout

```
mri_dataset_brain_cancer_oc/
├── avec_labels/        # labeled images grouped by class
└── sans_label/         # unlabeled study images
```

Place the dataset at the repository root (default assumption for the scripts).

## Data audit (Task 1)

Regenerate the exploratory audit assets (tables, plots, and notes):

```bash
python -m src.data_audit --data-dir mri_dataset_brain_cancer_oc --sample-size 64
```

Artifacts are written to `outputs/tables`, `outputs/figures`, and
`outputs/notes/data_audit.md`.

## Feature extraction (Task 2)

The `src.feature_extraction` module builds a deterministic preprocessing
pipeline (resize→center crop→tensor→ImageNet normalization) and extracts
512-D embeddings using a frozen `torchvision` ResNet-18 backbone. Run the full
inference pass with:

```bash
python -m src.feature_extraction \
  --data-dir mri_dataset_brain_cancer_oc \
  --device cpu            # or cuda if available \
  --batch-size 32
```

Key artifacts are produced under `outputs/`:

- `outputs/features/embeddings.npy` — NumPy array of [N, 512] features.
- `outputs/features/embeddings.csv` — table aligning file paths to embedding
  indices (ignored from Git by default).
- `outputs/features/metadata.json` — reproducibility metadata (backbone,
  transforms, dataset digest, sanity check statistics).
- `outputs/logs/feature_extraction.log` — run log.
- `outputs/notes/feature_summary.md` — short human-readable report with stats
  and nearest-neighbor spot checks.

Decode failures (if any) are logged and summarized in the markdown report so
that problematic files can be investigated separately.

## Semi-supervised training & comparison (Task 4)

Train the supervised baseline and semi-supervised pipeline with:

```bash
python -m src.semi_supervised_training \
  --strong-data-dir mri_dataset_brain_cancer_oc/avec_labels \
  --weak-data-dir mri_dataset_brain_cancer_oc/sans_label \
  --baseline-epochs 10 \
  --weak-pretrain-epochs 5 \
  --finetune-epochs 8
```

Key outputs (ignored by Git) are written under `outputs/`:

- `tables/results_comparison.csv` — Accuracy, precision, recall, F1, and wall-clock time for each model.
- `figures/train_curves_*.png`, `confusion_matrix_*.png`, `roc_curves.png` — visual diagnostics.
- `notes/training_history.json` — metrics per epoch plus data split indices.
- `models/*.pt` — best-performing checkpoints for local reuse.

A concise narrative summary and interpretation is committed at
`notes/training_report.md`.
