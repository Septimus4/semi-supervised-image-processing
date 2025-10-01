# Performance

Use these tips to maximize throughput on fast GPUs (e.g., RTX 5090) and multi-core CPUs.

- Batch size: Increase `--batch-size` until VRAM or training stability becomes a concern.
- DataLoader workers: Use `--num-workers` to parallelize CPU image decoding/augmentation. On a 32-thread CPU, 12–20 is typically effective.
- Pin memory: Enabled automatically when CUDA is available for faster host→device copies.
- CUDNN benchmarking vs determinism: The training script sets deterministic behavior by default via `set_seed`, which disables `torch.backends.cudnn.benchmark`. For extra speed (non-deterministic), you could enable benchmarking; consider exposing a flag if needed.
- Mixed precision (AMP): Not enabled by default; adding `torch.cuda.amp.autocast` + `GradScaler` can further accelerate training. This is a good next enhancement.
- Channel-last memory format: For CNNs, using `.to(memory_format=torch.channels_last)` can improve performance; not currently used.
- Persistent workers / prefetch: You may enable `persistent_workers=True` and tune `prefetch_factor` for additional pipeline efficiency.
