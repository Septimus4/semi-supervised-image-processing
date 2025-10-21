# Clustering Metrics Snapshot (updated)

- Standardization summary (mean |μ| / mean σ): labeled 0.154 / 0.948, unlabeled 0.011 / 1.002.
- PCA components to reach 90% explained variance: 145.
- K-Means sweep on PCA space (k=2–10): best at k=3 with silhouette 0.1023, ARI 0.4149, NMI 0.4036.
- DBSCAN with scope=labeled and auto-selected eps (via k-distance 98th percentile): best at eps≈24.293, min_samples=10 with silhouette 0.1093, ARI 0.0353, NMI 0.0828, noise_rate 0.14.
- Generated k-distance diagnostics: `outputs/figures/kdist_plot_labeled_ms{5,10,15}.png` and `outputs/figures/kdist_plot_labeled.png`.

- DBSCAN with scope=unlabeled and auto-selected eps: best at eps≈32.045, min_samples=5 with silhouette 0.2463 on the fitted subset and a very low noise rate (~0.07%). ARI/NMI are 0 by design for unlabeled scope (no ground truth).
- Additional diagnostics: `outputs/figures/kdist_plot_unlabeled_ms{5,10,15}.png` and `outputs/figures/kdist_plot_unlabeled.png`.

Notes:
- ARI/NMI are computed on the labeled subset only; silhouette is computed on the fitted subset for DBSCAN.
- Using labeled-only scope for DBSCAN avoids the all-noise collapse observed when fitting on all points; unlabeled points retain label -1.

Computed from `outputs/features/standardized_features.npz` on 2025-10-21; metrics sourced from `outputs/tables/metrics_clustering.csv` and summary matched to `outputs/notes/clustering_report.md`.
