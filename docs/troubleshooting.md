# Troubleshooting

- Missing dataset structure
  - Ensure `mri_dataset_brain_cancer_oc/avec_labels` and `mri_dataset_brain_cancer_oc/sans_label` exist as described in [Dataset](./dataset.md).

- Torch cannot find CUDA
  - Check `torch.cuda.is_available()` after activation. Install a CUDA-enabled torch build and verify NVIDIA drivers.

- Out of memory (OOM) on GPU
  - Reduce `--batch-size`. Ensure no other heavy GPU workloads are running.

- Unreadable images during feature extraction
  - The run will continue but log failures. Inspect `outputs/logs/feature_extraction.log` and the summary note.

- Slow training or data loading
  - Increase `--num-workers` and ensure the dataset is on fast storage (SSD/NVMe).

- Model download failures (torchvision weights)
  - Ensure Internet access for the first run or manually place the weights in the torch cache.
