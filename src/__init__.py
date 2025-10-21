"""Semi-supervised Image Processing — Overview

This package is structured to walk students through an end-to-end
image learning workflow on an MRI brain cancer dataset:

1) Data audit (``src.data_audit``)
	- Why: build a mental model of the data (sizes, modes, corrupt files).
	- What you learn: dataset hygiene and quick, repeatable diagnostics.

2) Feature extraction (``src.feature_extraction``)
	- Why: reuse a strong, general feature space for multiple tasks.
	- What you learn: deterministic transforms, pretrained backbones, and
	  artifact logging for reproducible experiments.

3) Unsupervised exploration (``src.clustering``)
	- Why: understand structure before labels; avoid leakage; pick metrics.
	- What you learn: PCA/t-SNE/UMAP trade-offs, K-Means/DBSCAN behavior,
	  internal vs external validation.

4) Semi-supervised training (``src.semi_supervised_training``)
	- Why: leverage abundant unlabeled data via pseudo-labeling.
	- What you learn: baselines, pseudo-label thresholds, early stopping,
	  target-recall threshold selection, and fair comparisons.

5) Threshold analysis (``src.threshold_sweep``)
	- Why: select operating points aligned with application goals (e.g.,
	  high recall screening).
	- What you learn: confusion-derived metrics and ROC/PR interpretation.

All modules write human-readable artifacts under ``outputs/`` so you can
compare runs and discuss trade-offs with evidence.
"""
