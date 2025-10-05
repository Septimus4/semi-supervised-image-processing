# Clustering Metrics Snapshot

- Standardization summary (mean |μ| / mean σ): labeled 0.154 / 0.948, unlabeled 0.011 / 1.002.
- PCA components to reach 90% explained variance: 145.
- K-Means sweep on PCA space (k=2–6):
  - k=2 → silhouette 0.100, ARI 0.379, NMI 0.362.
  - k=3 → silhouette 0.102, ARI 0.433, NMI 0.420.
  - k=4 → silhouette 0.099, ARI 0.341, NMI 0.356.
  - k=5 → silhouette 0.072, ARI 0.354, NMI 0.376.
  - k=6 → silhouette 0.072, ARI 0.332, NMI 0.380.
- DBSCAN trials (ε∈{0.7,1.2}, min_samples∈{5,10}) collapsed into all-noise assignments (noise rate ≈1.0), yielding undefined silhouette/ARI/NMI.

_Computed from `outputs/features/standardized_features.npz` on 2025-10-05 using scikit-learn 1.5.2._
