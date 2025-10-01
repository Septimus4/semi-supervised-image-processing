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
- `--device` (str: `auto|cpu|cuda`, default: `auto`)
- `--output-dir` (Path, default: `outputs`)
