"""
Root conftest.py — applies to all test sessions.

Caps torch/numpy thread pools so the full suite doesn't saturate all CPU cores
and freeze the machine. Also lowers process priority (nice) on Linux.
"""
import os

# Cap parallelism BEFORE any torch/numpy imports.
# Default: half the cores, minimum 2, so the system stays responsive.
_max_threads = str(max(2, os.cpu_count() // 2))

os.environ.setdefault("OMP_NUM_THREADS", _max_threads)
os.environ.setdefault("MKL_NUM_THREADS", _max_threads)
os.environ.setdefault("OPENBLAS_NUM_THREADS", _max_threads)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Apply torch thread cap if already imported (some fixtures import early)
try:
    import torch
    torch.set_num_threads(int(_max_threads))
    torch.set_num_interop_threads(2)
except Exception:
    pass

# Lower process priority on Linux so desktop stays usable
try:
    os.nice(10)
except (OSError, AttributeError):
    pass  # Windows or permission denied
