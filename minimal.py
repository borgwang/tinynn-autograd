import time

import numpy as np

from core.tensor import Tensor

np.random.seed(0)


BS = 512
idim = 512
odim = 512

data_x = np.random.normal(0, 1, (BS, idim)).astype(np.float32)
data_y = np.random.normal(0, 1, (BS, odim)).astype(np.float32)
data_w = np.random.normal(0, 1, (idim, odim)).astype(np.float32)

n_ep = 100

#@profile
def run_gpu():
    print("GPU")

    x = Tensor(data_x).gpu()
    y = Tensor(data_y).gpu()
    w = Tensor(data_w, requires_grad=True).gpu()

    t0 = time.time()
    for epoch in range(n_ep):
        w.zero_grad()
        pred = x @ w
        err = pred - y
        loss = (err ** 2).sum()
        loss.backward()
        w -= 0.0001 * w.grad
    t1 = time.time()
    print(f"GPU compute cost: {t1 - t0:.5f} s")
    print(f"loss check: {loss.values.get():.8f}")


#@profile
def run_cpu():
    print("CPU")
    x, y, w = data_x, data_y, data_w

    t0 = time.time()
    for epoch in range(n_ep):
        pred = x @ w
        err = pred - y
        loss = (err * err).sum()
        dw = x.T @ (2 * err)
        w -= 0.0001 * dw
    t1 = time.time()
    print(f"CPU compute cost: {t1 - t0:.3f}s")
    print(f"loss check: {loss:.8f}")

run_gpu()
run_cpu()

