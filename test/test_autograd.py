"""test unit for autograd"""

import runtime_path  # isort:skip

import numpy as np

from core.tensor import Tensor


def test_add_op():
    devices = ("gpu", )
    for device in devices:
        t1 = Tensor([1, 3, 5], requires_grad=True).to(device)
        t2 = Tensor([5, -2, -9], requires_grad=True).to(device)
        t3 = t1 + t2
        assert np.allclose(t3.numpy(), [6, 1, -4])
        t3.backward([2, 2, 2])
        assert np.allclose(t1.grad.numpy(), [2, 2, 2])
        assert np.allclose(t2.grad.numpy(), [2, 2, 2])

        # broadcast (2, 3) + (3,) -> (2, 3)
        t1 = Tensor([[1, 3, 5], [2, 3, 0]], requires_grad=True).to(device)
        t2 = Tensor([5, -2, -9], requires_grad=True).to(device)
        t3 = t1 + t2
        assert np.allclose(t3.numpy(), [[6, 1, -4], [7, 1, -9]])
        t3.backward([[1, 1, 1], [2, 2, 2]])
        assert np.allclose(t1.grad.numpy(), [[1, 1, 1], [2, 2, 2]])
        assert np.allclose(t2.grad.numpy(), [3, 3, 3])

        # broadcast (2, 3) + (1, 3) -> (2, 3)
        t1 = Tensor([[1, 3, 5], [2, 3, 0]], requires_grad=True).to(device)
        t2 = Tensor([[5, -2, -9]], requires_grad=True).to(device)
        t3 = t1 + t2
        assert np.allclose(t3.numpy(), [[6, 1, -4], [7, 1, -9]])
        t3.backward([[1, 1, 1], [2, 2, 2]])
        assert np.allclose(t1.grad.numpy(), [[1, 1, 1], [2, 2, 2]])
        assert np.allclose(t2.grad.numpy(), [3, 3, 3])

def test_mul_ops():
    devices = ("gpu",)
    for device in devices:
        t1 = Tensor([1, 3, 5], requires_grad=True).to(device)
        t2 = Tensor([5, -2, -9], requires_grad=True).to(device)
        t3 = t1 * t2
        assert np.allclose(t3.numpy(), [5, -6, -45])
        t3.backward([2, 2, 2])
        assert np.allclose(t1.grad.numpy(), [2*5, 2*(-2), 2*(-9)])
        assert np.allclose(t2.grad.numpy(), [2*1, 2*3, 2*5])


def test_div_ops():
    devices = ("gpu",)
    for device in devices:
        t1 = Tensor([1, 2, 5], requires_grad=True).to(device)
        t2 = Tensor([8, -2, -10], requires_grad=True).to(device)
        t3 = t1 / t2
        assert np.allclose(t3.numpy(), [0.125, -1, -0.5])
        t3.backward([1, 1, 1])
        assert np.allclose(t1.grad.numpy(), [0.125, -0.5, -0.1])
        assert np.allclose(t2.grad.numpy(), [-0.015625, -0.5, -0.05])

def test_pow_ops():
    devices = ("gpu",)
    for device in devices:
        t1 = Tensor([1, -3, 5], requires_grad=True).to(device)
        t2 = t1 ** 3
        assert np.allclose(t2.numpy(), [1, -27, 125])
        t2.backward([2, 2, 2])
        assert np.allclose(t1.grad.numpy(), [2 * 3 * 1 ** 2, 2 * 3 * (-3) ** 2, 2 * 3 * 5 ** 2])

def test_dot_ops():
    devices = ("gpu",)
    for device in devices:
        t1 = Tensor([[1, 3, 5], [5, -2, 9]], requires_grad=True).to(device)
        t2 = Tensor([[9, 8, 9, 7], [4, 0, 3, 0], [0, 8, 2, 7]], requires_grad=True).to(device)
        t3 = t1 @ t2
        assert np.allclose(t3.numpy(), [[21, 48, 28, 42], [37, 112, 57, 98]])
        t3.backward([[1, 2, 3, 4], [4, 3, 2, 1]])
        assert np.allclose(t1.grad.numpy(), [[80, 13, 50], [85, 22, 35]])
        assert np.allclose(t2.grad.numpy(), [[21, 17, 13, 9], [-5, 0, 5, 10], [41, 37, 33, 29]])


def test_sum_ops():
    devices = ("gpu",)
    for device in devices:
        t1 = Tensor([1, 3, 5], requires_grad=True).to(device)
        t2 = Tensor([5, -2, -9], requires_grad=True).to(device)
        t3 = (t1 + t2).sum()
        assert t3.numpy() == 3
        t3.backward(2)
        assert np.allclose(t1.grad.numpy(), [2, 2, 2])
        assert np.allclose(t2.grad.numpy(), [2, 2, 2])


def test_epx_ops():
    devices = ("gpu",)
    for device in devices:
        data = [1, 3, 4]
        t1 = Tensor(data, requires_grad=True).to(device)
        t2 = t1.exp()
        assert np.allclose(t2.numpy(), np.exp(data))
        t2.backward([1, 2, 3])
        assert np.allclose(t1.grad.numpy(), np.exp(data) * np.array([1, 2, 3]))


def test_neg_ops():
    devices = ("gpu",)
    for device in devices:
        t1 = Tensor([1, 3, 5], requires_grad=True).to(device)
        t2 = -t1
        assert np.allclose(t2.numpy(), [-1, -3, -5])
        t2.backward([1, 2, 3])
        assert np.allclose(t1.grad.numpy(), [-1, -2, -3])

def test_permute_ops():
    devices = ("gpu",)
    for device in devices:
        shape = [2, 4, 6]
        data = np.random.randn(*shape)
        t1 = Tensor(data, requires_grad=True).to(device)
        t2 = t1.T
        assert list(t2.shape) == shape[::-1]
        t2.backward(2 * np.ones(t2.shape))
        assert list(t1.grad.shape) == shape
        assert np.allclose(t1.grad.numpy(), 2 * np.ones(t1.shape))

def test_max_ops():
    devices = ("gpu",)
    for device in devices:
        t1 = Tensor([[1, 3, 5], [3, 7, -2]], requires_grad=True).to(device)
        t2 = t1.max()
        t3 = t1.max(axis=0)
        assert t2.numpy() == 7
        assert np.allclose(t3.numpy(), [3, 7, 5])
        t2.backward()
        assert np.allclose(t1.grad.numpy(), [[0, 0, 0], [0, 1, 0]])
        t1.zero_grad()
        t3.backward([1, 1, 1])
        assert np.allclose(t1.grad.numpy(), [[0, 0, 1], [1, 1, 0]])


def test_log_ops():
    devices = ("gpu",)
    for device in devices:
        data = np.array([1, 3, 5])
        t1 = Tensor(data, requires_grad=True).to(device)
        t2 = t1.log()
        assert np.allclose(t2.numpy(), np.log(data))

        grad = np.array([1, 2, 3])
        t2.backward(grad)
        assert np.allclose(t1.grad.numpy(), grad / np.array([1, 3, 5]))


def test_reshape_ops():
    devices = ("gpu",)
    for device in devices:
        t1 = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True).to(device)
        t2 = t1.reshape((6,))
        assert np.allclose(t2.numpy(), [1, 2, 3, 4, 5, 6])

        t2.backward(np.ones(6))
        assert np.allclose(t1.grad.numpy(), [[1, 1, 1], [1, 1, 1]])

