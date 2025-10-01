# Feature Extraction

This step computes fixed 512-D embeddings using a frozen `torchvision` ResNet-18 on deterministic ImageNet-style transforms.

Run (GPU recommended):
```bash
python -m src.feature_extraction \
  --data-dir mri_dataset_brain_cancer_oc \
  --device cuda \
  --batch-size 128 \
  --verbose
```

Key outputs:
- `outputs/features/embeddings.npy` — float32 array [N, 512]
- `outputs/features/embeddings.csv` — paths to embedding indices
- `outputs/features/metadata.json` — backbone, transforms, dataset digest, device
- `outputs/notes/feature_summary.md` — quick report with stats and NN spot-check

Notes:
- Grayscale inputs are converted to RGB using PIL and normalized with ImageNet stats.
- The module logs decode failures and continues; see `outputs/logs/feature_extraction.log`.
