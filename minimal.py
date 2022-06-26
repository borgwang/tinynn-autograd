import time

import numpy as np

from core.tensor import Tensor

np.random.seed(0)

BS = 512
idim = 512
odim = 512

"""
data_x = np.arange(12).reshape((2, 3, 2)).astype(np.float32)
data_y = np.ones((4, 2)).astype(np.float32)
print(data_x @ data_y.T)
x = Tensor(data_x).gpu()
y = Tensor(data_y).gpu()
pred = x @ y.T
print(pred.cpu())
"""

"""
data_x = np.arange(12).reshape(3, 4).astype(np.float32)
data_x = np.arange(12).astype(np.float32)
print(data_x.sum())
x = Tensor(data_x).gpu()
print(x.sum().cpu())
import pdb; pdb.set_trace()
"""

data_x = np.random.normal(0, 1, (BS, idim)).astype(np.float32)
data_y = np.random.normal(0, 1, (BS, odim)).astype(np.float32)
data_w = np.random.normal(0, 1, (idim, odim)).astype(np.float32)
data_b = np.zeros((1, odim)).astype(np.float32)

n_ep = 100

def run_gpu():
    print("GPU")

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

        #loss = (err * err).sum()
        #loss.backward()
        #w -= 0.0001 * w.grad
    t1 = time.time()
    print(f"GPU compute cost: {t1 - t0:.5f} s")
    print(f"err check: {err.values.to_cpu().sum():.8f}")
    #print(f"loss check: {loss.values.to_cpu():.8f}")


def run_cpu():
    print("CPU")
    x, y, w, b = data_x, data_y, data_w, data_b

    t0 = time.time()
    for epoch in range(n_ep):
        pred = x @ w + b
        err = pred - y
        #loss = (err * err).sum()
        #dw = x.T @ (2 * err)
        #w -= 0.0001 * dw
    t1 = time.time()
    print(f"CPU compute cost: {t1 - t0:.3f}s")
    print(f"err check: {err.sum():.8f}")
    #print(f"loss check: {loss:.8f}")

run_gpu()
run_cpu()

