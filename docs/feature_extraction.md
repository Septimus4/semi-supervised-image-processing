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
- The module logs decode failures and continues; see `outputs/logs/feature_extraction.log`.
- Input channels: no RGB conversion is performed here. The preprocessing expects images to already be 3‑channel RGB to match ImageNet normalization. If you introduce true single‑channel grayscale images later, either replicate the channel at load time or add a conditional conversion step upstream.
