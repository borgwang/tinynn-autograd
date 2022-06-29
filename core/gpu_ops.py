"""Tensor operations (with autograd context)"""

from functools import lru_cache

import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array


@lru_cache()
def cl_build(name, program):
    from core.tensor import CTX, QUEUE
    cl_kernel = cl.Program(CTX, program).build().__getattr__(name)
    return lambda *args: cl_kernel(QUEUE, *args)


def as_tensor(obj):
    from core.tensor import as_tensor
    return as_tensor(obj)


def build_binary_ops_tensor(ts1, ts2, grad_fn_ts1, grad_fn_ts2, values):
    requires_grad = ts1.requires_grad or ts2.requires_grad
    dependency = []
    if ts1.requires_grad:
        dependency.append(dict(tensor=ts1, grad_fn=grad_fn_ts1))
    if ts2.requires_grad:
        dependency.append(dict(tensor=ts2, grad_fn=grad_fn_ts2))
    tensor_cls = ts1.__class__
    return tensor_cls(values, requires_grad, dependency)


def build_unary_ops_tensor(ts, grad_fn, values):
    requires_grad = ts.requires_grad
    dependency = []
    if ts.requires_grad:
        dependency.append(dict(tensor=ts, grad_fn=grad_fn))
    tensor_cls = ts.__class__
    return tensor_cls(values, requires_grad, dependency)


def add_(ts1, ts2):
    values = ts1.values + ts2.values
    grad_fn_ts1 = grad_fn_ts2 = lambda g: g
    return build_binary_ops_tensor(ts1, ts2, grad_fn_ts1, grad_fn_ts2, values)


def sub_(ts1, ts2):
    values = ts1.values - ts2.values
    grad_fn_ts1 = lambda g: g
    grad_fn_ts2 = lambda g: -g
    return build_binary_ops_tensor(ts1, ts2, grad_fn_ts1, grad_fn_ts2, values)


def mul_(ts1, ts2):
    values = ts1.values * ts2.values
    grad_fn_ts1 = lambda g: g * ts2.values
    grad_fn_ts2 = lambda g: g * ts1.values
    return build_binary_ops_tensor(ts1, ts2, grad_fn_ts1, grad_fn_ts2, values)


def div_(ts1, ts2):
    values = ts1.values / ts2.values

    # c = a / b
    # D_c / D_a = 1 / b
    # D_c / D_b = -a / b**2
    def grad_fn_ts1(grad):
        grad = grad / ts2.values
        for _ in range(grad.ndim - ts1.values.ndim):
            grad = grad.sum(axis=0)
        for i, dim in enumerate(ts1.shape):
            if dim == 1:
                grad = grad.sum(axis=i, keepdims=True)
        return grad

    def grad_fn_ts2(grad):
        grad = -grad * ts1.values / ts2.values ** 2
        for _ in range(grad.ndim - ts2.values.ndim):
            grad = grad.sum(axis=0)
        for i, dim in enumerate(ts2.shape):
            if dim == 1:
                grad = grad.sum(axis=i, keepdims=True)
        return grad

    return build_binary_ops_tensor(
        ts1, ts2, grad_fn_ts1, grad_fn_ts2, values)


def pow_(ts1, ts2):
    values = ts1.values ** ts2.values

    # c = a ** b
    # D_c / D_a = b * a ** (b-1)
    # D_c / D_b = ln(a) * a ** b
    def grad_fn_ts1(grad):
        return grad * (ts2.values * ts1.values ** (ts2.values - np.ones((), dtype=np.float32)))

    def grad_fn_ts2(grad):
        # TODO
        grad = grad * (np.log(ts1.values) * values)
        for _ in range(grad.ndim - ts2.values.ndim):
            grad = grad.sum(axis=0)
        for i, dim in enumerate(ts2.shape):
            if dim == 1:
                grad = grad.sum(axis=i, keepdims=True)
        return grad

    return build_binary_ops_tensor(
        ts1, ts2, grad_fn_ts1, grad_fn_ts2, values)


def contiguous_transpose_op(ts):
    # NOTE: input is a tensor because we need the shape attribute from tensor instance.
    from core.tensor import QUEUE
    assert ts.values.flags.c_contiguous, "Array must be contiguous before transpose_op!"
    length = np.prod(ts.values.shape)
    values = cl_array.empty(QUEUE, (length,), dtype=np.float32)
    op = cl_build("transpose_op", """
    __kernel void transpose_op(const int M, const int N,
        __global const float *A, __global float *B) {
      const int i = get_global_id(0);
      B[i] = A[(i % M) * N + i / M];
    }
    """)
    # TODO: support 3D/4D transpose
    M, N = np.int32(ts.values.shape[0]), np.int32(ts.values.shape[1])
    op((length,), None, M, N, ts.values.data, values.data)
    return values.reshape((N, M))


def matmul_(ts1, ts2):
    values = ts1.values @ ts2.values
    def grad_fn_ts1(grad):
        return grad @ t2.values.T
    grad_fn_ts1 = lambda g: g @ ts2.values.T
    grad_fn_ts2 = lambda g: ts1.values.T @ g
    return build_binary_ops_tensor(ts1, ts2, grad_fn_ts1, grad_fn_ts2, values)


def maximum_(ts1, ts2):
    values = np.maximum(ts1.values, ts2.values)

    def grad_fn_ts1(grad):
        grad = grad * (ts1.values >= ts2.values)
        for _ in range(grad.ndim - ts1.values.ndim):
            grad = grad.sum(axis=0)
        for i, dim in enumerate(ts1.shape):
            if dim == 1:
                grad = grad.sum(axis=i, keepdims=True)
        return grad

    def grad_fn_ts2(grad):
        grad = grad * (ts2.values > ts1.values)
        for _ in range(grad.ndim - ts2.values.ndim):
            grad = grad.sum(axis=0)
        for i, dim in enumerate(ts2.shape):
            if dim == 1:
                grad = grad.sum(axis=i, keepdims=True)
        return grad

    return build_binary_ops_tensor(
        ts1, ts2, grad_fn_ts1, grad_fn_ts2, values)


def minimum_(ts1, ts2):
    values = np.minimum(ts1.values, ts2.values)

    def grad_fn_ts1(grad):
        grad = grad * (ts1.values <= ts2.values)
        for _ in range(grad.ndim - ts1.values.ndim):
            grad = grad.sum(axis=0)
        for i, dim in enumerate(ts1.shape):
            if dim == 1:
                grad = grad.sum(axis=i, keepdims=True)
        return grad

    def grad_fn_ts2(grad):
        grad = grad * (ts2.values < ts1.values)
        for _ in range(grad.ndim - ts2.values.ndim):
            grad = grad.sum(axis=0)
        for i, dim in enumerate(ts2.shape):
            if dim == 1:
                grad = grad.sum(axis=i, keepdims=True)
        return grad

    return build_binary_ops_tensor(
        ts1, ts2, grad_fn_ts1, grad_fn_ts2, values)


def exp_(ts):
    values = np.exp(ts.values)

    def grad_fn(grad):
        return values * grad

    return build_unary_ops_tensor(ts, grad_fn, values)


def max_(ts, axis, keepdims):
    values = ts.values.max(axis=axis, keepdims=keepdims)

    def grad_fn(grad):
        return grad * (ts.values.max(axis=axis, keepdims=1) == ts.values)

    return build_unary_ops_tensor(ts, grad_fn, values)


def min_(ts, axis=None):
    values = np.min(ts.values, axis=axis)

    def grad_fn(grad):
        return grad * (ts.values.min(axis=axis, keepdims=1) == ts.values)

    return build_unary_ops_tensor(ts, grad_fn, values)


def log_(ts):
    values = np.log(ts.values)
    def grad_fn(grad):
        return grad / ts.values
    return build_unary_ops_tensor(ts, grad_fn, values)


def sum_(ts, axis, keepdims):
    values = ts.values.sum(axis, keepdims)
    if axis is not None:  # TODO
        repeat = ts.values.shape[axis]
    def grad_fn(grad):
        if axis is None:
            #return grad * cl_array.zeros(QUEUE, ts.shape, dtype=np.float32) + 1.0
            return grad * grad.__class__.ones(ts.shape)
        else:
            grad = np.expand_dims(grad, axis)
            grad = np.repeat(grad, repeat, axis)
        return grad
    return build_unary_ops_tensor(ts, grad_fn, values)


def transpose_(ts, axes=None):
    if axes is None:
        assert len(ts.values.shape) == 2
        axes = (1, 0)

    values = ts.values.transpose(axes)
    if axes is None:
        axes = reversed(range(ts.values.ndim))
    axes = list(axes)
    # recover to original shape
    def grad_fn(grad):
        return grad.transpose(np.argsort(axes))
    return build_unary_ops_tensor(ts, grad_fn, values)


def getitem_(ts, key):
    values = ts.values[key]

    def grad_fn(grad):
        recover_grad = np.zeros_like(ts.values)
        recover_grad[key] = grad
        return recover_grad

    return build_unary_ops_tensor(ts, grad_fn, values)


def neg_(ts):
    values = -ts.values

    def grad_fn(grad):
        return -grad

    return build_unary_ops_tensor(ts, grad_fn, values)


def reshape_(ts, newshape):
    oldshape = ts.values.shape
    values = ts.values.reshape(newshape)

    def grad_fn(grad):
        return grad.reshape(oldshape)

    return build_unary_ops_tensor(ts, grad_fn, values)


def pad_(ts, pad_width, mode):
    values = np.pad(ts.values, pad_width=pad_width, mode=mode)
    slices = list()
    for size, (before, after) in zip(values.shape, pad_width):
        slices.append(slice(before, size-after))

    def grad_fn(grad):
        return grad[tuple(slices)]

    return build_unary_ops_tensor(ts, grad_fn, values)


def flatten_(ts):
    shape = ts.shape
    values = ts.values.ravel()

    def grad_fn(grad):
        return grad.reshape(shape)
    return build_unary_ops_tensor(ts, grad_fn, values)


def clip_(ts, min, max):
    values = ts.values.clip(min, max)

    mask = np.ones(ts.shape, dtype=bool)
    if min is not None:
        mask &= ts.values >= min
    if max is not None:
        mask &= ts.values <= max

    def grad_fn(grad):
        return grad * mask
    return build_unary_ops_tensor(ts, grad_fn, values)


#def max(obj, axis=None):
#    return max_(as_tensor(obj), axis=axis)


def maximum(obj1, obj2):
    return maximum_(as_tensor(obj1), as_tensor(obj2))


def minimum(obj1, obj2):
    return minimum_(as_tensor(obj1), as_tensor(obj2))


def exp(obj):
    return exp_(as_tensor(obj))


def sum(obj, axis=None):
    return sum_(as_tensor(obj), axis=axis)


def log(obj):
    return log_(as_tensor(obj))


def reshape(obj, newshape):
    return reshape_(as_tensor(obj), newshape)


def pad(obj, pad_width, mode="constant"):
    return pad_(as_tensor(obj), pad_width, mode=mode)


def flatten(obj):
    return flatten_(as_tensor(obj))


def clip(obj, min=None, max=None):
    return clip_(as_tensor(obj), min, max)


def random_normal(loc, scale, size):
    pass

