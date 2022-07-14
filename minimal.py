import argparse
import time

import numpy as np
from core.tensor import Tensor

np.random.seed(0)

BS = 2**6
idim = 2**8
odim = 2**6

data_x = np.random.normal(0, 1, (BS, idim)).astype(np.float32)
data_y = np.random.normal(0, 1, (BS, odim)).astype(np.float32)
data_w = np.random.normal(0, 1, (idim, odim)).astype(np.float32)
data_b = np.zeros((1, odim)).astype(np.float32)

n_ep = 10

def run_gpu():
    print("---- GPU -----")
    x = Tensor(data_x).to(device)
    y = Tensor(data_y).to(device)
    w = Tensor(data_w, requires_grad=True, name="w").to(device)
    b = Tensor(data_b, requires_grad=True, name="b").to(device)

    t0 = time.time()
    for epoch in range(n_ep):
        w.zero_grad()
        b.zero_grad()
        pred = x @ w + b
        err = pred - y
        loss = (err**2).sum()
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="gpu", type=str)
    args = parser.parse_args()
    device = args.device
    run_gpu()
    run_cpu()

