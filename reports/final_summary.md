# Semi-Supervised MRI Pipeline — Synthesis and Scale-Up Plan

## Executive Summary
- Audited a mixed-labeled MRI dataset (100 labeled / 1,406 unlabeled) and confirmed consistent RGB 512×512 imagery with no decode failures, establishing a reliable foundation for downstream automation.【F:outputs/notes/data_audit.md†L5-L29】
- Extracted 512-D ResNet18 embeddings (0.039 s/image on CPU, zero failures) and validated standardized feature quality for both labeled and unlabeled pools.【F:outputs/notes/feature_summary.md†L1-L18】
- PCA + K-Means clustering using 145 principal components achieved a silhouette of 0.102 and ARI/NMI of 0.433/0.420 (k=3), confirming separability while highlighting moderate label noise; DBSCAN was unstable at comparable scales.【F:notes/clustering_metrics.md†L1-L12】
- Semi-supervised fine-tuning trailed the supervised baseline on test F1 (0.84 vs. 0.90), indicating pseudo-label precision as the critical lever before scaling to millions of images.【F:notes/training_report.md†L5-L37】
- Scaling to 4M images under €5k is feasible via GPU-accelerated batch embedding, active labeling triage, and lightweight retraining while aggressively caching embeddings and using spot-priced compute; detailed plan and budget are provided below.【F:notes/scaleup_plan.md†L3-L48】【F:tables/scaleup_estimate.csv†L1-L6】

## Workflow Recap (Tasks 1–4)
### Task 1 — Data Audit
- Structure: `avec_labels/{cancer, normal}` (50 each) and `sans_label` (1,406 images).【F:outputs/notes/data_audit.md†L5-L16】
- Uniform modality: RGB 512×512; suggested grayscale conversion optional for modeling alignment.【F:outputs/notes/data_audit.md†L11-L29】
- Bottlenecks: storage footprint (~38 KB/image) manageable; labeling ratio 1:14 drives semi-supervised choices.

### Task 2 — Feature Extraction
- Backbone: torchvision ResNet18 (ImageNet weights) with deterministic resize → center-crop → normalization pipeline.【F:outputs/notes/feature_summary.md†L1-L12】
- Performance: 1,506 embeddings with mean per-image latency 0.039 s (CPU), zero decode failures, and healthy feature dispersion (|mean|≈0.885, σ≈0.582).【F:outputs/notes/feature_summary.md†L8-L18】
- Outputs: 512-D float32 embeddings, metadata JSON, nearest-neighbor sanity checks.【F:outputs/notes/feature_summary.md†L20-L36】

### Task 3 — Dimensionality Reduction & Clustering
- Standardization retained near-zero mean / unit variance (mean|μ|<0.16 labeled, <0.02 unlabeled).【F:notes/clustering_metrics.md†L1-L4】
- PCA required 145 components to explain 90% variance, indicating rich intra-class diversity.【F:notes/clustering_metrics.md†L1-L5】
- K-Means sweep (k=2–6) peaked at k=3 with silhouette 0.102, ARI 0.433, NMI 0.420, uncovering partially aligned structure but moderate overlap between clinical classes.【F:notes/clustering_metrics.md†L4-L10】
- DBSCAN with ε≤1.2 collapsed into all-noise clusters, confirming density-based methods need adaptive metrics or precomputed graphs for scale.【F:notes/clustering_metrics.md†L11-L12】

### Task 4 — Semi-Supervised Training
- Baseline supervised ResNet18: Accuracy/Precision/Recall/F1 = 0.90/0.90/0.90/0.90 in 75 s of training.【F:notes/training_report.md†L19-L27】
- Semi-supervised variant (pseudo-labels ≥0.6 confidence): Accuracy 0.85, F1 0.84 with higher training cost (196 s) and lower recall, emphasizing pseudo-label calibration before scale-up.【F:notes/training_report.md†L21-L37】
- Diagnostics showed higher false negatives and noisier convergence for pseudo-labeled pre-training.【F:notes/training_report.md†L29-L37】

## Strengths, Limitations, and Bottlenecks
- **Strengths:** Deterministic preprocessing, reproducible feature extraction, and modular scripts for each stage. Embedding storage (~2 KB/image) is compact, enabling large-scale caching.【F:outputs/notes/feature_summary.md†L1-L18】
- **Limitations:** Pseudo-label noise suppresses gains; clustering quality moderate; CPU-only extraction throughput limits scalability (0.039 s/image ⇒ 43.6 GPU-equivalent hours for 4M images after acceleration).【F:outputs/notes/feature_summary.md†L8-L18】【F:notes/scaleup_plan.md†L12-L20】
- **Bottlenecks:** Manual labeling (clinician time), feature inference throughput, I/O for millions of files, configuration tracking across reruns.

## Scaling Feasibility Analysis (4M Images)
### Throughput & Compute
- Target pipeline: GPU-resident ResNet18 inference at 10× CPU throughput (≈0.004 s/image). This yields ~4.5 GPU-days for 4M images; parallelizing across four spot T4/L4 instances reduces wall-clock to ~27 hours.【F:notes/scaleup_plan.md†L12-L20】
- Fine-tuning reuse: freeze backbone; only train linear/low-rank adapters on labeled + curated pseudo-labeled batches. Estimated 30 GPU hours/iteration with two iterations across rollout phases.【F:notes/scaleup_plan.md†L26-L28】

### Storage & Memory
- Raw MRI JPEG/PNG ingestion at ~40 KB each ⇒ 160 GB for 4M slices (compressible via object storage lifecycle policies). Embedding cache: 4M × 512 × 2 bytes (float16) ≈ 3.9 GB; logistic metadata stored in Parquet.<br>
- Stream ingestion using PyTorch `DataLoader` with multiprocessing and on-the-fly conversion prevents RAM saturation.【F:notes/scaleup_plan.md†L12-L20】

### Labeling & Quality Control
- Adopt active learning: rank uncertainty on embedding classifier to request ~3,000 expert labels (0.075% of pool). Combined with consensus review and pseudo-label self-training doubles effective labeled set while staying within €1,500 labeling allowance.【F:notes/scaleup_plan.md†L22-L37】
- Maintain pseudo-label precision via confidence calibration, iterative filtering, and validation slices logged to MLflow.

### Budget Summary
A €5k budget envelope is allocated as:
- **Compute (€3,000):** Spot T4/L4 GPU hours for embedding + fine-tuning, with fallback CPU instances for orchestration.【F:tables/scaleup_estimate.csv†L1-L6】
- **Storage (€500):** S3-compatible object store (160 GB raw + 5 GB embeddings + replication) using infrequent-access tiers and lifecycle policies.【F:tables/scaleup_estimate.csv†L2-L5】
- **Labeling (€1,500):** Clinician review of 3,000 samples at €0.50 each plus QA buffer.【F:tables/scaleup_estimate.csv†L3-L5】
- **Overhead (€0–€500 reserve):** Holds for CI/CD, monitoring, and contingency (absorbed within compute bucket by throttling spot usage).【F:tables/scaleup_estimate.csv†L1-L6】

## Deployment & MLOps Recommendations
1. **Modular Orchestration:** Airflow or Prefect DAG with stages — data ingestion → preprocessing → embedding cache → clustering triage → active labeling → retraining → evaluation. Each node writes metrics/config JSON to versioned storage for traceability.【F:notes/scaleup_plan.md†L8-L48】
2. **Reusable Embeddings:** Persist float16 embeddings and indexes (FAISS/Annoy) to serve downstream search, clustering, and bootstrapped labeling without re-running the backbone.【F:notes/scaleup_plan.md†L12-L27】
3. **CI/CD & Experiment Tracking:** GitHub Actions for lint/tests, DVC for data pointers, MLflow for experiment metadata, and TorchServe/FastAPI containers for inference endpoints.【F:notes/scaleup_plan.md†L30-L48】
4. **Monitoring:** Automate metric dumps (JSON) and drift dashboards; set alerts on pseudo-label confidence shifts and GPU utilization to throttle spending.
5. **Phased Rollout:** Pilot on 100k images to validate throughput and labeling strategy before scaling to full 4M dataset, using the same orchestration templates and budget guardrails.【F:notes/scaleup_plan.md†L44-L48】

## Next Steps
- Implement active learning loop + pseudo-label QA to lift semi-supervised gains prior to scale.
- Containerize inference & data prep for reproducible deployment on managed Kubernetes with autoscaling spot nodes.
- Finalize monitoring dashboards and cost watchdogs before full-scale ingestion.
