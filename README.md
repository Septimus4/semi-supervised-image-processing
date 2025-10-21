# Semi-Supervised Image Processing

Utilities for auditing and preprocessing the MRI brain cancer dataset used in
semi-supervised experiments.

➡️ Full documentation is available in `docs/`. Start here: [docs/index.md](docs/index.md)

## Environment setup

```bash
pip install -e .
```

This installs PyTorch, torchvision, and the supporting scientific Python stack
needed by the audit, feature extraction, and training scripts.

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
pipeline (resize→tensor→ImageNet normalization; no RGB conversion) and extracts
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

## Training (baseline and semi-supervised)

Train the semi-supervised pipeline (includes a supervised baseline for comparison):

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

Run a supervised-only baseline:
```bash
python -m src.supervised_training \
  --strong-data-dir mri_dataset_brain_cancer_oc/avec_labels \
  --baseline-epochs 10 \
  --batch-size 32 \
  --device cuda
```

## Calibrate the decision threshold

The training script automatically selects a probability cutoff on the validation split to meet your deployment goals:

- It targets a desired recall (`--target-recall`) and optionally enforces `--min-precision` and/or `--max-fpr`.
- Among thresholds that satisfy the constraints, it picks the largest one (to reduce false positives).
- If constraints can’t be met, it falls back to maximizing F-beta (`--f-beta`, default 2.0, recall-weighted).

Recommended run (example, semi-supervised):

```bash
python -m src.semi_supervised_training \
  --strong-data-dir mri_dataset_brain_cancer_oc/avec_labels \
  --weak-data-dir mri_dataset_brain_cancer_oc/sans_label \
  --target-recall 0.98 \
  --min-precision 0.60
```

Where to find the chosen threshold and diagnostics:

- Tables: `outputs/tables/results_comparison.csv` and `outputs/tables/results_comparison_detailed.csv` (look for `threshold`, `threshold_policy`, `target_recall`, and constraints columns).
- Curves: `outputs/figures/pr_curves.png` and `outputs/figures/roc_curves.png` to inspect trade-offs.

Optional manual sweep (for analysis only):

```bash
python -m src.threshold_sweep \
  --strong-data-dir mri_dataset_brain_cancer_oc/avec_labels \
  --model semi \
  --device cuda
```

This writes `outputs/tables/threshold_sweep_{baseline|semi}.csv` and prints the highest threshold achieving TPR≈1.0 on the test split. Calibrate on validation for deployment; use the test sweep for understanding, not tuning.

## Documentation

- Overview and navigation: [docs/index.md](docs/README.md)
- Setup: [docs/setup.md](docs/setup.md)
- Dataset layout: [docs/dataset.md](docs/dataset.md)
- Data audit: [docs/data_audit.md](docs/data_audit.md)
- Feature extraction: [docs/feature_extraction.md](docs/feature_extraction.md)
- Training: [docs/training.md](docs/training.md)
- CLI reference: [docs/cli_reference.md](docs/cli_reference.md)
- Performance tips: [docs/performance.md](docs/performance.md)
- Troubleshooting: [docs/troubleshooting.md](docs/troubleshooting.md)
- Architecture: [docs/architecture.md](docs/architecture.md)
- Reproducibility: [docs/reproducibility.md](docs/reproducibility.md)
 - Evaluation and operating points: [docs/evaluation.md](docs/evaluation.md)
