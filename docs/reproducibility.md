# Reproducibility

- Seeds: Training sets random seeds across Python, NumPy, and Torch to favor determinism.
- CUDNN deterministic mode: Enabled; `benchmark` disabled.
- Dataset digest: Feature extraction computes a SHA-256 digest of discovered files and sizes in `outputs/features/metadata.json`.
- Metadata: Embedding config, transforms, and device info recorded in `outputs/features/metadata.json`.
- Training history: Curves and split indices are saved to `outputs/notes/training_history.json`.
- Environment capture: Consider recording `pip freeze` or `uv pip compile` outputs alongside runs for full reproducibility.
