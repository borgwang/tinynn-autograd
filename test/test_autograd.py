"""test unit for autograd"""

import runtime_path  # isort:skip

import numpy as np

import core.ops as ops
from core.tensor import Tensor


def test_add_op():
    t1 = Tensor([1, 3, 5], requires_grad=True)
    t2 = Tensor([5, -2, -9], requires_grad=True)
    t3 = t1 + t2
    assert t3.values.tolist() == [6, 1, -4]
    t3.backward([2, 2, 2])

    assert t1.grad.tolist() == [2, 2, 2]
    assert t2.grad.tolist() == [2, 2, 2]

    # broadcast (2, 3) + (3,) -> (2, 3)
    t1 = Tensor([[1, 3, 5], [2, 3, 0]], requires_grad=True)
    t2 = Tensor([5, -2, -9], requires_grad=True)
    t3 = t1 + t2
    assert t3.values.tolist() == [[6, 1, -4], [7, 1, -9]]
    t3.backward([[1, 1, 1], [2, 2, 2]])
    assert t1.grad.tolist() == [[1, 1, 1], [2, 2, 2]]
    assert t2.grad.tolist() == [3, 3, 3]

    # broadcast (2, 3) + (1, 3) -> (2, 3)
    t1 = Tensor([[1, 3, 5], [2, 3, 0]], requires_grad=True)
    t2 = Tensor([[5, -2, -9]], requires_grad=True)
    t3 = t1 + t2
    assert t3.values.tolist() == [[6, 1, -4], [7, 1, -9]]
    t3.backward([[1, 1, 1], [2, 2, 2]])
    assert t1.grad.tolist() == [[1, 1, 1], [2, 2, 2]]
    assert t2.grad.tolist() == [[3, 3, 3]]


def test_mul_ops():
    t1 = Tensor([1, 3, 5], requires_grad=True)
    t2 = Tensor([5, -2, -9], requires_grad=True)
    t3 = t1 * t2
    assert t3.values.tolist() == [5, -6, -45]
    t3.backward([2, 2, 2])

    assert t1.grad.tolist() == [2*5, 2*(-2), 2*(-9)]
    assert t2.grad.tolist() == [2*1, 2*3, 2*5]


def test_div_ops():
    t1 = Tensor([1, 2, 5], requires_grad=True)
    t2 = Tensor([8, -2, -10], requires_grad=True)
    t3 = t1 / t2
    assert t3.values.tolist() == [0.125, -1, -0.5]
    t3.backward([1, 1, 1])

    assert t1.grad.tolist() == [0.125, -0.5, -0.1]
    assert t2.grad.tolist() == [-0.015625, -0.5, -0.05]


def test_pow_ops():
    t1 = Tensor([1, -3, 5], requires_grad=True)
    t2 = t1 ** 3
    assert t2.values.tolist() == [1, -27, 125]
    t2.backward([2, 2, 2])
    assert t1.grad.tolist() == [2 * 3 * 1 ** 2, 2 * 3 * (-3) ** 2, 2 * 3 * 5 ** 2]


def test_dot_ops():
    t1 = Tensor([[1, 3, 5], [5, -2, 9]], requires_grad=True)
    t2 = Tensor([[9, 8, 9, 7], [4, 0, 3, 0], [0, 8, 2, 7]], requires_grad=True)
    t3 = t1 @ t2
    assert t3.values.tolist() == [[21, 48, 28, 42], [37, 112, 57, 98]]
    t3.backward([[1, 2, 3, 4], [4, 3, 2, 1]])
    assert t1.grad.tolist() == [[80, 13, 50], [85, 22, 35]]
    assert t2.grad.tolist() == [[21, 17, 13, 9], [-5, 0, 5, 10], [41, 37, 33, 29]]


def test_sum_ops():
    t1 = Tensor([1, 3, 5], requires_grad=True)
    t2 = Tensor([5, -2, -9], requires_grad=True)
    t3 = (t1 + t2).sum()
    assert t3.values == 3
    t3.backward(2)
    assert t1.grad.tolist() == [2, 2, 2]
    assert t2.grad.tolist() == [2, 2, 2]


def test_epx_ops():
    t1 = Tensor([1, 3, 5], requires_grad=True)
    t2 = ops.exp(t1)
    assert t2.values.tolist() == np.exp(t1.values).tolist()

    t2.backward([1, 2, 3])
    assert t1.grad.tolist() == (np.exp(t1.values) * np.array([1, 2, 3])).tolist()


def test_neg_ops():
    t1 = Tensor([1, 3, 5], requires_grad=True)
    t2 = -t1
    assert t2.values.tolist() == [-1, -3, -5]

    t2.backward([1, 2, 3])
    assert t1.grad.tolist() == [-1, -2, -3]


def test_minimal_nn():
    x = Tensor(np.random.normal(0, 1.0, (100, 3)))
    y = x * 3.14 + 30

    w1 = Tensor(np.random.normal(0, 1.0, (3, 3)), requires_grad=True)
    b1 = Tensor(np.random.normal(0, 1.0, 3), requires_grad=True)

    previous_loss = 1e10
    for _ in range(100):
        w1.zero_grad()
        b1.zero_grad()
        predicted = x @ w1 + b1
        err = predicted - y
        loss = (err ** 2).sum()
        loss.backward()
        w1 -= 0.001 * w1.grad
        b1 -= 0.001 * b1.grad
        assert loss.values < previous_loss
        previous_loss = loss.values


def test_maximum_ops():
    t1 = Tensor([1, 3, 5], requires_grad=True)
    t2 = Tensor([5, -2, 9], requires_grad=True)
    t3 = ops.maximum_(t1, t2)

    assert t3.values.tolist() == [5, 3, 9]
    t3.backward([1, 2, 1])
    assert t1.grad.tolist() == [0, 2, 0]
    assert t2.grad.tolist() == [1, 0, 1]


def test_minimum_ops():
    t1 = Tensor([1, 3, 5], requires_grad=True)
    t2 = Tensor([5, -2, 9], requires_grad=True)
    t3 = ops.minimum_(t1, t2)

    assert t3.values.tolist() == [1, -2, 5]
    t3.backward([1, 2, 1])
    assert t1.grad.tolist() == [1, 0, 1]
    assert t2.grad.tolist() == [0, 2, 0]


def test_transpose_ops():
    shape = [2, 4, 6]
    data = np.random.randn(*shape)
    t1 = Tensor(data, requires_grad=True)
    t2 = t1.T
    assert list(t2.shape) == shape[::-1]

    t2.backward(np.ones_like(t2.values))
    assert list(t1.grad.shape) == shape

    t2 = t1.transpose((2, 0, 1))
    assert list(t2.shape) == [6, 2, 4]

    t2.backward(np.ones_like(t2.values))
    assert list(t1.grad.shape) == shape


def test_max_ops():
    t1 = Tensor([[1, 3, 5], [3, 7, -2]], requires_grad=True)
    t2 = ops.max(t1, axis=None)
    t3 = ops.max(t1, axis=0)
    assert t2.values == 7
    assert t3.values.tolist() == [3, 7, 5]

    t2.backward()
    assert t1.grad.tolist() == [[0, 0, 0], [0, 1, 0]]
    t1.zero_grad()
    t3.backward([1, 1, 1])
    assert t1.grad.tolist() == [[0, 0, 1], [1, 1, 0]]


def test_log_ops():
    t1 = Tensor([1, 3, 5], requires_grad=True)
    t2 = ops.log(t1)
    assert t2.values.tolist() == np.log(t1.values).tolist()

    grad = np.array([1, 2, 3])
    t2.backward(grad)
    assert t1.grad.tolist() == (grad / np.array([1, 3, 5])).tolist()


def test_reshape_ops():
    t1 = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)
    t2 = ops.reshape(t1, (6,))
    assert t2.values.tolist() == [1, 2, 3, 4, 5, 6]

    t2.backward(np.ones(6))
    assert t1.grad.tolist() == [[1, 1, 1], [1, 1, 1]]


def test_pad_ops():
    t1 = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)
    pad_width = [(1, 0), (1, 0)]
    t2 = ops.pad(t1, pad_width)
    assert t2.values.tolist() == [[0, 0, 0, 0], [0, 1, 2, 3], [0, 4, 5, 6]]

    t2.backward(np.ones_like(t2.values))
    assert t1.grad.shape == t1.shape
    assert t1.grad.tolist() == [[1, 1, 1], [1, 1, 1]]


def test_flatten_ops():
    t1 = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)
    t2 = ops.flatten(t1)
    assert t2.values.tolist() == [1, 2, 3, 4, 5, 6]

    t2.backward(np.ones_like(t2.values))
    assert t1.grad.shape == t1.shape
    assert t1.grad.tolist() == [[1, 1, 1], [1, 1, 1]]


def test_clip_ops():
    t1 = Tensor([1, -3, 5], requires_grad=True)
    t2 = ops.clip(t1, 0)
    assert t2.values.tolist() == [1, 0, 5]

    grad = np.array([1, 2, 3])
    t2.backward(grad)
    assert t1.grad.tolist() == [1, 0, 3]
