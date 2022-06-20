import time

import numpy as np

from core.tensor import Tensor


BS = 512
idim = 256
odim = 128

data_x = np.random.normal(0, 1, (BS, idim)).astype(np.float32)
data_y = np.random.normal(0, 1, (BS, odim)).astype(np.float32)
data_w = np.random.normal(0, 1, (idim, odim)).astype(np.float32)

n_ep = 3000


def run_gpu():
    print("GPU")
    t0 = time.time()

    x = Tensor(data_x).gpu()
    y = Tensor(data_y).gpu()
    #w = Tensor(data_w, requires_grad=True).gpu()
    w = Tensor(data_w).gpu()

    t1 = time.time()
    print(f"to GPU cost: {t1 - t0:.5f} s")

    for epoch in range(n_ep):
        # zero gradients
        #w.zero_grad()
        # forward
        pred = x @ w
        #err = pred - y
        #loss = (err * err).sum()
        # backward
        #loss.backward()
        # updata parameters
        #w -= 0.0001 * w.grad

    t2 = time.time()
    print(f"GPU compute cost: {t2 - t1:.5f} s")
    print(f"GPU total cost: {t2 - t0:.5f} s")
    print(f"value check: {pred.values.get().sum():.8f}")

    print("CPU")


def run_cpu():
    x, y, w = data_x, data_y, data_w

    t0 = time.time()
    for epoch in range(n_ep):
        pred = x @ w
        #err = pred - y
        #loss = (err * err).sum()
        #dw = x.T @ (2 * err)
        #w -= 0.0001 * dw
    t1 = time.time()
    print(f"CPU compute cost: {t1 - t0:.3f}s")
    print(f"value check: {pred.sum():.8f}")

run_gpu()
run_cpu()

