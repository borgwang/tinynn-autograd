import os

DEBUG = int(os.getenv("DEBUG", "0"))
GRAPH = int(os.getenv("GRAPH", "0"))
BACKEND = os.getenv("BACKEND", "opencl")
assert BACKEND in ("numpy", "opencl", "cuda"), f"backend {BACKEND} not supported!"

