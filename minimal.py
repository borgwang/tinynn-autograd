import time
import sys

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

#data_x = np.arange(12).reshape(3, 4).astype(np.float32)
def test_sum_1d():
    data_x = np.arange(2**16).astype(np.float32)
    a = data_x.sum()
    b = Tensor(data_x).gpu().sum()
    assert np.allclose(a, b.cpu())

def test_sum_2d():
    data_x = np.random.randint(-5, 6, (2**8, 2**8)).astype(np.float32)
    a = data_x.sum()
    b = Tensor(data_x).gpu().sum(axis=0)
    assert np.allclose(a, b.cpu())


def test_sum_2d_axis_0():
    #data_x = np.arange(2**12).astype(np.float32)
    data_x = np.random.randint(-5, 6, (2**16, 10)).astype(np.float32)
    a = data_x.sum(axis=0)
    b = Tensor(data_x).gpu().sum(axis=0)
    assert np.allclose(a, b.cpu())
    print("test_sum_2d_axis_0 pass")

def test_sum_2d_axis_1():
    #data_x = np.arange(2**12).astype(np.float32)
    data_x = np.random.randint(-5, 6, (10, 2**16)).astype(np.float32)
    a = data_x.sum(axis=1)
    b = Tensor(data_x).gpu().sum(axis=1)
    assert np.allclose(a, b.cpu())
    print("test_sum_2d_axis_1 pass")

def test_sum_3d_axis_0():
    #data_x = np.arange(2**12).astype(np.float32)
    data_x = np.random.randint(-5, 6, (2**16, 3, 3)).astype(np.float32)
    a = data_x.sum(axis=0)
    b = Tensor(data_x).gpu().sum(axis=0)
    assert np.allclose(a, b.cpu())
    print("test_sum_3d_axis_0 pass")


def test_max_3d_axis_0():
    #data_x = np.arange(2**12).astype(np.float32)
    data_x = np.random.randint(-5, 6, (2**16, 3, 3)).astype(np.float32)
    a = data_x.max(axis=0)
    b = Tensor(data_x).gpu().max(axis=0)
    assert np.allclose(a, b.cpu())
    print("test_sum_3d_axis_0 pass")

#test_sum_1d()
test_sum_2d_axis_0()
test_sum_2d_axis_1()
test_sum_3d_axis_0()
test_max_3d_axis_0()
sys.exit()

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

