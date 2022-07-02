import time
import sys

import numpy as np

from core.tensor import Tensor

np.random.seed(0)

from core.ndarray import GPUArray

"""
def check_array(myarr, nparr):
    assert myarr.shape == nparr.shape  # shape
    assert myarr.dtype == nparr.dtype  # dtype
    # strides
    np_strides = tuple(s // myarr.dtype().itemsize for s in nparr.strides)
    assert myarr.strides == np_strides
    # contiguousness
    assert myarr.c_contiguous == nparr.flags.c_contiguous
    assert myarr.f_contiguous == nparr.flags.f_contiguous
    # values
    assert np.allclose(myarr.numpy(), nparr)

shape = (2**8,)
nparr = np.arange(np.prod(shape)).reshape(shape).astype(np.float32)
arr = GPUArray(nparr)
check_array(arr.sum(), nparr.sum())
print("pass")
sys.exit()
"""

BS = 2**8
idim = 2**12
odim = 2**8

data_x = np.random.normal(0, 1, (BS, idim)).astype(np.float32)
data_y = np.random.normal(0, 1, (BS, odim)).astype(np.float32)
data_w = np.random.normal(0, 1, (idim, odim)).astype(np.float32)
data_b = np.zeros((1, odim)).astype(np.float32)

n_ep = 10

def run_gpu():
    print("---- GPU -----")
    x = Tensor(data_x).gpu()
    y = Tensor(data_y).gpu()
    w = Tensor(data_w, requires_grad=True).gpu()
    b = Tensor(data_b, requires_grad=True).gpu()

    t0 = time.time()
    for epoch in range(n_ep):
        w.zero_grad()
        b.zero_grad()
        pred = x @ w + b
        err = pred - y
        loss = (err * err).sum()
        loss.backward()
        w -= 0.0001 * w.grad
        b -= 0.0001 * b.grad

    t1 = time.time()
    print(f"GPU compute cost: {t1 - t0:.5f} s")
    print(f"err check: {err.values.numpy().sum():.8f}")
    print(f"loss check: {loss.values.numpy():.8f}")


def run_cpu():
    print("---- CPU -----")
    x, y, w, b = data_x, data_y, data_w, data_b

    t0 = time.time()
    for epoch in range(n_ep):
        pred = x @ w + b
        err = pred - y
        loss = (err * err).sum()
        dw = x.T @ (2 * err)
        db = (2 * err).sum(axis=0, keepdims=True)
        w -= 0.0001 * dw
        b -= 0.0001 * db
    t1 = time.time()
    print(f"CPU compute cost: {t1 - t0:.3f}s")
    print(f"err check: {err.sum():.8f}")
    print(f"loss check: {loss:.8f}")

run_gpu()
run_cpu()

