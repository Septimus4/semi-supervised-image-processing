# CLI Reference

## src.data_audit
- `--data-dir` (Path, default: `mri_dataset_brain_cancer_oc`)
- `--sample-size` (int, default: 64)
- `--seed` (int, default: 42)

## src.feature_extraction
- `--data-dir` (Path, default: `mri_dataset_brain_cancer_oc`)
- `--device` (str, default: `cuda` if available else `cpu`)
- `--batch-size` (int, default: 32)
- `--verbose` (flag)

## src.semi_supervised_training
- `--strong-data-dir` (Path, required)
- `--weak-data-dir` (Path, required)
- `--batch-size` (int, default: 16)
- `--val-split` (float, default: 0.2)
- `--test-split` (float, default: 0.2)
- `--seed` (int, default: 42)
- `--image-size` (int, default: 224)
- `--num-workers` (int, default: 2)
- `--baseline-epochs` (int, default: 10)
- `--weak-pretrain-epochs` (int, default: 5)
- `--finetune-epochs` (int, default: 8)
- `--pseudo-threshold` (float, default: 0.7)
- `--learning-rate` (float, default: 1e-4)
- `--weight-decay` (float, default: 1e-4)
- `--early-stopping` (int, default: 3)
- `--positive-class` (str, default: `cancer`) — name of the folder for the positive class
- `--target-recall` (float, optional) — target recall for threshold selection on validation set; if omitted, threshold tuning is disabled and argmax predictions are used
- `--min-precision` (float, optional) — minimum precision constraint for threshold selection
- `--max-fpr` (float, optional) — maximum false positive rate constraint for threshold selection
- `--f-beta` (float, default: 2.0) — fallback selection favors recall when constraints can’t be met (beta > 1)
- `--device` (str: `auto|cpu|cuda`, default: `auto`)
- `--output-dir` (Path, default: `outputs`)
- `--unlabeled-cohort-csv` (Path, optional) — CSV with a `path` column to filter the weak pool (e.g., from DBSCAN non-noise cohort)

## src.threshold_sweep
- `--strong-data-dir` (Path, required)
- `--output-dir` (Path, default: `outputs`)
- `--model` (str: `baseline|semi`, default: `semi`)
- `--positive-class` (str, default: `cancer`)
- `--device` (str, default: `cuda` — falls back to `cpu` if CUDA is unavailable)
- `--num-workers` (int, default: 4)

## src.clustering
- `--features-npz` (Path, required) — path to standardized feature bundle (`standardized_features.npz`)
- `--output-root` (Path, default: `outputs`)
- `--variance-target` (float, default: 0.9) — PCA explained-variance target for clustering space
- `--tsne-dim` (int, default: 50) — PCA components fed into t-SNE/UMAP
- `--tsne-perplexities` (float, multiple, default: `[10.0, 30.0, 50.0]`)
- `--umap-neighbors` (int, multiple, default: `[15, 30, 50]`)
- `--umap-min-dist` (float, multiple, default: `[0.0, 0.1]`)
- `--kmeans-range` (int, multiple, default: `2–10`)
- `--kmeans-n-init` (int, default: 10)
- `--dbscan-eps` (float, multiple, default: `[0.5, 0.75, 1.0, 1.25]`)
- `--dbscan-min-samples` (int, multiple, default: `[5, 10, 15]`)
- `--dbscan-scope` (str: `all|labeled|unlabeled`, default: `all`) — fit DBSCAN on all points, labeled-only, or unlabeled-only; labels for non-fitted points are set to -1
- `--dbscan-auto` (flag) — auto-select `eps` via the 98th percentile of the k-distance curve for each `min_samples`; also saves k-distance plots under `outputs/figures/`
- `--seed` (int, default: 42)
- `--log-level` (str: `DEBUG|INFO|WARNING|ERROR`, default: `INFO`)

## src.export_unlabeled_cohort
- `--assignments` (Path, default: `outputs/tables/cluster_assignments.csv`)
- `--method` (str: `dbscan|kmeans`, default: `dbscan`)
- `--cluster-id` (int, optional) — select a specific cluster; when omitted with DBSCAN, all non-noise samples are exported
- `--output` (Path, default: `outputs/tables/unlabeled_cohort.csv`)

## src.standardize_features
- `--embeddings-npy` (Path, default: `outputs/features/embeddings.npy`)
- `--embeddings-csv` (Path, default: `outputs/features/embeddings.csv`)
- `--output-npz` (Path, default: `outputs/features/standardized_features.npz`)
- `--log-level` (str: `DEBUG|INFO|WARNING|ERROR`, default: `INFO`)
