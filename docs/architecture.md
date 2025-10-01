# Architecture

```
repo/
├── src/
│   ├── data_audit.py              # dataset sampling + plots + notes
│   ├── feature_extraction.py      # deterministic transforms + ResNet18 features
│   └── semi_supervised_training.py# baseline + pseudo-label + fine-tune pipeline
├── outputs/                       # generated artifacts (ignored by Git)
├── notes/                         # hand-written reports
├── README.md
└── pyproject.toml
```

- `data_audit.py` produces quick insights into the dataset’s contents and quality.
- `feature_extraction.py` generates 512-D embeddings with a frozen ResNet-18; metadata is stored for reproducibility.
- `semi_supervised_training.py` implements a two-stage semi-supervised approach: pseudo-labeling then fine-tuning, with a supervised baseline for comparison.
