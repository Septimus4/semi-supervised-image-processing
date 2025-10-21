# Architecture

```
repo/
├── src/
│   ├── data_audit.py                 # dataset sampling + plots + notes
│   ├── feature_extraction.py         # deterministic transforms + ResNet18 embeddings
│   ├── supervised_training.py        # CLI wrapper → training/supervised.run_supervised
│   ├── semi_supervised_training.py   # CLI wrapper → training/semi_supervised.run_pipeline
│   ├── threshold_sweep.py            # evaluate thresholds on the test split
│   ├── standardize_features.py       # normalize embeddings (PCA/scale bundle)
│   ├── clustering.py                 # PCA/t-SNE/UMAP + KMeans/DBSCAN exploration
│   └── training/
│       ├── common.py                 # shared config, transforms, datasets, loops, plots, thresholds
│       ├── supervised.py             # baseline training/eval pipeline
│       └── semi_supervised.py        # baseline → pseudo-label → weak-pretrain → fine-tune
├── outputs/                          # generated artifacts (ignored by Git)
├── reports/                          # curated summaries for sharing
├── docs/                             # living documentation
├── README.md
└── pyproject.toml
```

- `data_audit.py` produces quick insights into the dataset’s contents and quality.
- `feature_extraction.py` generates 512-D embeddings with a frozen ResNet-18; metadata is stored for reproducibility.
- `training/common.py` centralizes reusable building blocks: reproducibility, transforms, datasets, balanced sampling, model creation, training/eval loops, plots, and threshold selection.
- `training/supervised.py` implements the supervised baseline; invoked via `src.supervised_training`.
- `training/semi_supervised.py` implements the semi-supervised pipeline; invoked via `src.semi_supervised_training`.
- `threshold_sweep.py` runs a post-hoc sweep over decision thresholds on the test set.
