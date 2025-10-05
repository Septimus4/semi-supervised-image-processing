# Scale-Up Plan Notes

## Objectives
- Process 4,000,000 MRI slices under a €5,000 operational budget.
- Preserve reproducibility across ingestion, embedding, clustering, and semi-supervised retraining stages.
- Maintain or exceed supervised baseline performance while leveraging pseudo-labels safely.

## Pipeline Automation
1. **Data Ingestion**
   - Sync raw MRI assets to S3-compatible object storage (versioned buckets).
   - Maintain manifest CSV/Parquet with checksums for deterministic splits.
2. **Preprocessing & Embedding**
   - Pre-resize images to 256 px on ingestion to avoid repeated decoding cost.
   - Use PyTorch `DataLoader` with 8 workers and pinned memory; convert to grayscale-on-the-fly if clinical requirement emerges.
   - Batch inference on spot GPUs (NVIDIA T4/L4) with mixed precision to achieve <5 ms/image throughput.
   - Throughput math: 4M images × 0.004 s ≈ 4.5 GPU-days (≈108 GPU-hours); current CPU rate (0.039 s/image) would require ≈173 GPU-hours equivalent, so GPU batching yields ~9.8× speedup.
   - Persist float16 embeddings + metadata to object storage and register location with DVC.
3. **Clustering & Selection**
   - Run PCA (90% variance) once per ingest batch; cache transformation matrix.
   - Execute K-Means at k∈{2,3,4} for drift monitoring; log ARI/NMI on labeled subset to MLflow.
   - Trigger active learning request when ARI drops >10% from baseline.
4. **Labeling Loop**
   - Serve clinicians with web-based triage (top-entropy + diversity sampling).
   - Target 3,000 expert annotations total, refreshed quarterly.
   - Promote pseudo-labels only when confidence ≥0.8 and consensus with embedding nearest-neighbour majority.
5. **Retraining**
   - Fine-tune classification head every 500k images ingested.
   - Use two-stage schedule: frozen backbone (5 epochs) → last two blocks unfrozen (3 epochs) with discriminative LR.

## Cost Guardrails
- **Compute:** Cap GPU spend at €3,000 by:
  - Reserving 200 GPU hours on spot T4 (€0.35/h) for embedding (4 parallel nodes × 12 h batches × 3 cycles).
  - Allocating 120 GPU hours on A10/T4 (€0.5/h) for retraining and evaluation.
  - Auto-shutdown idle nodes via orchestration heartbeat.
- **Storage:** 160 GB raw + 5 GB embeddings stored on Wasabi/Backblaze (~€0.005/GB/month) ≈ €0.8/day; budget €500/year with replication and transfer overhead.
- **Labeling:** Contract rate €0.50 per case with €0.05 QA buffer ⇒ €1.65k maximum; enforce budget by limiting active learning queue length.
- **Contingency:** 10% of total spend kept as credit reserve for retries or scaling spikes.

## Monitoring & QA
- Track ingestion throughput, GPU utilization, and queue sizes in Prometheus/Grafana dashboards.
- Validate pseudo-label quality using hold-out labeled slices each iteration; require F1 ≥0.88 before promoting new pseudo-label batches.
- Nightly CI job runs smoke tests on 1k-sample subset to ensure reproducibility of embeddings and clustering metrics.

## Deployment Milestones
1. **Pilot (Week 0–2):** Run pipeline on 100k images; finalize cost telemetry and storage settings.
2. **Scale (Week 3–6):** Expand to 1M images; integrate active learning; begin clinician labeling.
3. **Full Rollout (Week 7–10):** Process remaining 3M images in four 750k-image waves; retrain after each wave.
4. **Stabilization (Week 11+):** Transition to maintenance cadence (monthly retraining, weekly drift checks).
