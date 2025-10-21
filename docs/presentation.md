# Semi-Supervised MRI Tumor Detection — Project Presentation

Goal: Detect brain tumors with high recall while controlling reviewer workload using a semi-supervised approach that leverages unlabeled images.

---

## 1) Dataset and Business Context

- Source: Company MRI dataset (brain) with mixed labeled (`avec_labels/`) and unlabeled (`sans_label/`) pools.
- Scale: ~1.5k images in this iteration; plan scalable path to 4M images within a €5k budget.
- Objective: Catch as many tumors as possible (recall-first) without overwhelming doctors (control false positives).
- Compliance: Follows confidentiality constraints; ImageFolder layout for labeled data.

---

## 2) Data Cleaning and Quality Assurance

- Decoding & integrity checks
  - Why: Broken or unreadable files cause crashes and contaminate statistics.
  - How/Impact: Logged decode failures; excluded unreadable files from downstream steps to keep data consistent and runs reproducible.
  - See `outputs/notes/feature_summary.md` and `outputs/logs/*.log`.
- Handling missing values / outliers
  - Why: Outliers distort distributions and can bias models and cluster structure.
  - How/Impact: No tabular NA by design; image-level failures documented/excluded. Probed outliers via embeddings (PCA/t-SNE/UMAP) and nearest neighbors to flag anomalies for curation.
- Homogeneity of sources
  - Why: Different scanners/sites create domain shift in intensity/contrast/resolution.
  - How/Impact: Standardized preprocessing (resize→ImageNet normalization) harmonizes inputs; grayscale→RGB conversion aligns with ImageNet-pretrained expectations.
- Audit artifacts
  - Why: Quick visibility into dataset quality avoids sunk time later.
  - How/Impact: `src/data_audit.py` writes histograms, grids, directory summaries; see `outputs/figures/*` and `outputs/notes/data_audit.md`.

---

## 3) Bias Reduction and Risk Controls

- Class imbalance mitigation
  - Why: The labeled pool here is roughly balanced (50/50), but individual train splits can drift, and the large unlabeled pool likely has a different prevalence. A naïve trainer exposed to an imbalanced split may overfit the majority class, increasing FNs.
  - How/Impact: Use a class-balanced sampler on the supervised train split to equalize class exposure per minibatch. This guards against split-level imbalance and preserves recall for the positive class without changing labels.
- Source heterogeneity
  - Why: Images may come from different scanners/sites with varying contrast, resolution, and preprocessing.
  - How/Impact: Deterministic transforms and ImageNet mean/std normalization align channel statistics expected by the pretrained backbone, reducing domain shift and improving transfer stability.
- Outliers/inconsistent samples
  - Why: Corrupt files or atypical acquisitions can skew training and clustering, creating misleading patterns.
  - How/Impact: Visualize embeddings with t-SNE/UMAP and inspect nearest neighbors to flag anomalies for removal or review, keeping the training signal clean.
- Label noise (pseudo-labels)
  - Why: Pseudo-labels on the unlabeled pool can be wrong; low-confidence labels introduce noise that hurts learning.
  - How/Impact: Apply a confidence threshold when generating pseudo-labels: higher thresholds favor precision (cleaner labels) at the cost of coverage; we balance this to expand training data while limiting noise.

---

## 4) Code Readability and Maintainability

- Extensive inline comments across `src/` (neutral, rationale-focused).
  - Why: Reviewer/stakeholder handoff and future maintenance require clarity.
  - How/Impact: Comments explain choices (e.g., 224 sizing, LR schedule) to reduce onboarding time and errors.
- Modular functions: data prep, training loops, evaluation utilities, plotting.
  - Why: Modular code is testable and replaceable.
  - How/Impact: Enables targeted improvements (swap backbone, change sampler) without ripple effects.
- Reproducible runs: fixed seeds, saved splits/metrics/history, editable install with dev extras.
  - Why: Scientific validity and defensibility.
  - How/Impact: Same seed → same splits; saved artifacts enable audit and comparison.
- Artifacts: models, figures, tables, logs, notes saved under `outputs/`.
  - Why: Centralized artifacts accelerate review and debugging.
  - How/Impact: One-stop location for QA and presentation sourcing.

---

## 5) Data Transformations and Preparation

- Preprocessing pipeline
  - Resize to 224×224 (aligns with ResNet-18 ImageNet pretraining), center crop behavior via direct resize here.
  - ToTensor + ImageNet mean/std normalization.
- Why 224?
  - Why: Matching ImageNet training scale preserves receptive field assumptions.
  - How/Impact: Balances detail vs VRAM/throughput; reduces covariate shift for the pretrained backbone.
- Filtering/normalization
  - Why: Clustering is distance-sensitive; features on different scales skew distances.
  - How/Impact: Standardize embeddings (z-score) before clustering to ensure fair distance metrics.

---

## 6) Output Verification & Sanity Checks

- Embeddings: 512-D ResNet-18 features saved as `.npy`/`.csv`, with `metadata.json`.
  - Why: Downstream steps depend on these; we need traceability.
  - How/Impact: Include backbone, transforms, dataset digest in metadata for auditability.
- Sanity checks: nearest-neighbor probes and distribution summaries captured in `feature_summary.md`.
  - Why: Catch degenerate embeddings (e.g., all zeros) or mislabeled neighbors.
  - How/Impact: Quick qualitative validation that features encode meaningful similarity.
- Visual checks: PCA / t-SNE / UMAP plots to confirm structure and separability.
  - Why: Human-in-the-loop validation of cluster structure.
  - How/Impact: Guides clustering choices and identifies anomalies.

---

## 7) Clustering Experiments

- Methods tried: K-Means (k sweep), DBSCAN (epsilon/min_samples sweep).
- Evaluation: Adjusted Rand Index (ARI), Normalized Mutual Information (NMI) when labels available.
- Insights: K-Means k≈3 best in this run; DBSCAN tended to mark many points as noise for tested params.
- Deliverables: `outputs/tables/*clustering*.csv`, `outputs/figures/*pca|tsne|umap*.png`.
  - Why: Understand latent structure, discover cohorts, and surface potential mislabeled/outlier samples.
  - How/Impact: K-Means provides stable partitions for reporting; DBSCAN highlights noise and dense clusters.

---

## 8) Semi-Supervised Justification & Metrics

- Baseline vs Semi
  - Baseline: supervised training on labeled data only.
  - Semi: baseline → pseudo-label weak pool → pretrain head → fine-tune on strong labels.
- Operating point (argmax vs thresholded)
  - Argmax = default 0.5 cutoff; good for accuracy, not recall-first.
  - Thresholded = choose validation-calibrated threshold for target recall with constraints (min-precision / max FPR).
- Metrics used: accuracy, precision, recall, F1, ROC AUC, PR AP, confusion matrices.
  - Why: Capture both ranking (AUC/AP) and point decisions (precision/recall/F1) under class imbalance.
  - How/Impact: Aligns evaluation with clinical objective: high recall, bounded false positives.
- Evidence: `outputs/tables/results_comparison.csv` and detailed CSV/figures.

---

## 9) Hyperparameter Tuning

- Optimizer: AdamW (stable generalization on vision backbones).
- LR schedule: ReduceLROnPlateau (val loss), early stopping to avoid overfitting/plateaus.
- Key knobs: learning rate, weight decay, batch size, epochs, pseudo-label confidence threshold.
- Validation: best weights tracked by val loss; curves in `train_curves_*.png`.
  - Why: Avoid overfitting and local minima.
  - How/Impact: ReduceLROnPlateau + early stopping recover plateauing training and stop when no improvement.

---

## 10) Threshold Calibration for Deployment

- Policy: recall-first with constraints.
  - Example: target_recall=0.98, min_precision=0.60, optional max_fpr.
  - Choose the largest threshold satisfying constraints (fewer FPs).
- Recommended threshold (current run, semi model): 0.5913
  - Validation: recall 1.00, precision 0.91, FPR 0.10
  - Test (reference): accuracy 0.90, precision 0.90, recall 0.90, F1 0.90
- Triage CSV: `outputs/tables/unlabeled_predictions_semi.csv` (path, prob_positive, flagged)
  - Why: Operationalizes review flow and workload planning.
  - How/Impact: Sorted list lets clinicians prioritize the most confident positives first.

---

## 11) Deployment at Scale (4M images, €5,000)

- Hardware budget assumptions
  - €5k can provision: e.g., 1–2 modest GPUs (used RTX 3090/4090) for a month or short-term cloud GPU credits; or a CPU cluster of 16–32 cores with spot instances.
- Throughput planning
  - Feature extraction: ~150–250 img/s/GPU (AMP on) or ~15–30 img/s/CPU (estimate; verify on site). For 4M images: 4M / 200 img/s ≈ 5.6 hours/GPU; add I/O overhead and retries.
  - Batch inference + CSV triage for unlabeled images.
- Storage & I/O
  - Keep embeddings (`~2GB`/million at 512-D float32; halve with float16). For 4M: ~8GB (fp16) to ~16GB (fp32).
- Pipeline design
  - Sharded processing (per directory/day/site) with resumable jobs; log failures; per-shard checksums.
  - Deterministic preprocessing; version lock model, transforms, and thresholds in an `operating_point.json`.
- Cost controls
  - Prefer GPU spot instances; schedule off-peak; cache datasets locally; stream from object storage with prefetch.
- Ops/Monitoring
  - Metrics: images/min, error rate, cache hit, GPU util, mean confidence; sampling-based QA.
  - Why: Ensure throughput, quality, and cost don’t drift in production.
  - How/Impact: Early warning on data drift and performance regressions; enables capacity planning.

---

## 12) Link Technical Choices to Business Constraints

- Data volume: feature extraction enables cheap clustering/audits before labeling. Thresholding controls reviewer workload.
- Budget: pre-trained backbone reduces training cost/time; AMP reduces compute cost.
- Processing time: deterministic, parallelizable pipeline; resumable shards; balanced batch sizes.
- Risk: recall-first thresholding minimizes missed tumors; min-precision/max-FPR caps false alarms.
  - Why: Clinical risk is asymmetric (missed positives more costly than false alarms).
  - How/Impact: Operating point encodes business tolerance into model decisions.

---

## 13) Defense Talking Points

- Why semi-supervised? Gains from unlabeled data; apples-to-apples comparisons (same splits, metrics).
- Why ResNet-18/224? Fast, strong transfer, low VRAM; 256→224 standardization.
- Why AdamW + ReduceLROnPlateau + early stopping? Stable convergence, fewer wasted epochs.
- Why thresholding policy? Aligns with clinical priorities (miss fewer cases) while controlling workload.
- Calibration & validation discipline: validate thresholds on val split; keep test for final reporting.
- Reproducibility: seeds, saved splits, history, checkpoints, manifest.

---

## 14) Consistency Checks

- Notebook/results consistency
  - Metrics in tables match figures (training curves, PR/ROC, confusion matrices).
  - Threshold recorded in results tables/notes; same used to generate triage CSV.
- Re-run reproducibility
  - Re-running with the same seed reproduces splits and metrics within expected tolerance.
  - Why: Confidence in results and repeatability under audit.
  - How/Impact: Facilitates comparisons across experiments and regulatory reviews.

---

## 15) Next Steps

- Optional probability calibration (temperature scaling) to improve expected FP/FN estimates.
- Add a CLI to standardize embeddings to NPZ (quality-of-life for clustering).
- Lightweight CI to catch regressions (lint, type-check, tiny smoke tests).

---

## Appendix A — How to Run

- Data audit:
  ```bash
  python -m src.data_audit --data-dir mri_dataset_brain_cancer_oc --sample-size 64
  ```
- Feature extraction:
  ```bash
  python -m src.feature_extraction \
    --data-dir mri_dataset_brain_cancer_oc \
    --device cuda \
    --batch-size 64
  ```
- Train baseline + semi and auto-calibrate threshold:
  ```bash
  python -m src.semi_supervised_training \
    --strong-data-dir mri_dataset_brain_cancer_oc/avec_labels \
    --weak-data-dir mri_dataset_brain_cancer_oc/sans_label \
    --target-recall 0.98 \
    --min-precision 0.60
  ```
- Manual threshold sweep (analysis only):
  ```bash
  python -m src.threshold_sweep \
    --strong-data-dir mri_dataset_brain_cancer_oc/avec_labels \
    --model semi \
    --device cuda
  ```

---

## Appendix B — Key Artifacts

- Tables: `outputs/tables/results_comparison.csv`, `outputs/tables/results_comparison_detailed.csv`
- Figures: `outputs/figures/train_curves_*.png`, `outputs/figures/pr_curves.png`, `outputs/figures/roc_curves.png`, `outputs/figures/confusion_matrix_*.png`
- Notes: `outputs/notes/training_history.json`, `outputs/notes/data_audit.md`, `outputs/notes/feature_summary.md`
- Models: `outputs/models/*.pt`
- Triage CSV: `outputs/tables/unlabeled_predictions_semi.csv`
