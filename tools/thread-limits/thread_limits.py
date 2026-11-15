"""Cap BLAS/OpenMP thread pools before numpy/torch/sklearn load.

Imported automatically at interpreter startup via ``thread_limits.pth``.
NumPy's bundled OpenBLAS is built for at most 128 threads and crashes
("too many memory regions") on bigger machines, e.g. the 384-core JupyterHub
nodes. Values already set in the environment are left untouched, so you can
override them, e.g. ``OMP_NUM_THREADS=8 uv run ...``.
"""
import os

MAX_THREADS = 32

_n = str(min(MAX_THREADS, os.cpu_count() or 1))
for _var in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_var, _n)
