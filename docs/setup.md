# Setup

This project supports a fast, reproducible setup using uv (recommended) or standard pip.

## Prerequisites
- Python 3.9+
- NVIDIA GPU with CUDA drivers (optional but recommended)

## Option A — uv (recommended)
```bash
# Create and activate a virtual environment
uv venv .venv
source .venv/bin/activate

# Install the project in editable mode
uv pip install -e .
```

## Option B — pip
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Verify PyTorch and CUDA
```bash
python - <<'PY'
import torch
print('torch', torch.__version__)
print('cuda_available', torch.cuda.is_available())
if torch.cuda.is_available():
    print('device0', torch.cuda.get_device_name(0))
PY
```

If CUDA is not available but expected, ensure your NVIDIA driver and CUDA runtime are installed and compatible with the installed torch build.
