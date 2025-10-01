# Data Audit

The audit script samples files, extracts metadata, saves histograms and a sample grid, and writes a markdown report.

Run:
```bash
python -m src.data_audit --data-dir mri_dataset_brain_cancer_oc --sample-size 64
```

Artifacts:
- `outputs/tables/image_summary.csv` — per-sample metadata
- `outputs/tables/directory_summary.csv` — counts by bucket/subdir
- `outputs/figures/*.png` — sample grid and histograms
- `outputs/notes/data_audit.md` — human-readable notes

Common flags:
- `--data-dir` path to dataset root (default: `mri_dataset_brain_cancer_oc`)
- `--sample-size` number of files to inspect (default: 64)
- `--seed` for deterministic sampling (default: 42)
