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


def unary_op(name, a, shape):
    from core.tensor import QUEUE
    ret = cl_array.empty(QUEUE, a.shape, dtype=np.float32)
    op_mapping = {
        "neg": "-a",
    }
    code = op_mapping[name]
    unary_op = cl_build("unary_op", """
    __kernel void unary_op(
        __global const float4 *a_g, __global float4 *res_g) {
      int gid = get_global_id(0);
      float4 a = a_g[gid];
      res_g[gid] = """ + code + """;
    }
    """)
    unary_op(shape, None, a.data, ret.data)
    return ret


def binary_op(name, a, b, shape):
    from core.tensor import QUEUE
    ret = cl_array.empty(QUEUE, a.shape, dtype=np.float32)
    op_mapping = {
        "add": "a+b",
        "sub": "a-b",
        "div": "a/b",
        "mul": "a*b",
        "pow": "power(a,b)"
    }
    binary_op = cl_build("binary_op", """
    __kernel void binary_op(
        __global const float4 *a_g, __global const float4 *b_g, __global float4 *res_g) {
      int gid = get_global_id(0);
      float4 a = a_g[gid], b = b_g[gid];
      res_g[gid] = """ + op_mapping[name] + """;
    }
    """)
    binary_op(shape, None, a.data, b.data, ret.data)
    return ret

def as_tensor(obj):
    # avoid looping import
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
    gpu = isinstance(values, cl_array.Array)
    return tensor_cls(values, requires_grad, dependency, gpu=gpu)


def build_unary_ops_tensor(ts, grad_fn, values):
    requires_grad = ts.requires_grad
    dependency = []
    if ts.requires_grad:
        dependency.append(dict(tensor=ts, grad_fn=grad_fn))
    tensor_cls = ts.__class__
    gpu = isinstance(values, cl_array.Array)
    return tensor_cls(values, requires_grad, dependency, gpu=gpu)


def add_(ts1, ts2):
    #values = binary_op("add", ts1.values, ts2.values, ts1.shape)
    values = ts1.values + ts2.values
    # c = a + b
    # D_c / D_a = 1.0
    # D_c / D_b = 1.0
    # also need to handle broadcasting
    def grad_fn_ts1(grad):
        return grad

    def grad_fn_ts2(grad):
        return grad

    return build_binary_ops_tensor(ts1, ts2, grad_fn_ts1, grad_fn_ts2, values)


def sub_(ts1, ts2):
    return ts1 + (-ts2)


def mul_(ts1, ts2):
    values = binary_op("mul", ts1.values, ts2.values, ts1.shape)

    # c = a * b
    # D_c / D_a = b
    # D_c / D_b = a
    def grad_fn_ts1(grad):
        grad_values = binary_op("mul", grad.values, ts2.values, grad.shape)
        grad = grad.__class__(grad_values, shape=grad.shape, gpu=True)
        return grad

    def grad_fn_ts2(grad):
        #grad = grad * ts1.values
        grad_values = binary_op("mul", grad.values, ts1.values, grad.shape)
        grad = grad.__class__(grad_values, shape=grad.shape, gpu=True)
        return grad

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
        grad = grad * ts2.values * ts1.values ** (ts2.values - 1)
        for _ in range(grad.ndim - ts1.values.ndim):
            grad = grad.sum(axis=0)

        for i, dim in enumerate(ts1.shape):
            if dim == 1:
                grad = grad.sum(axis=i, keepdims=True)
        return grad

    def grad_fn_ts2(grad):
        grad = grad * (np.log(ts1.values) * values)
        for _ in range(grad.ndim - ts2.values.ndim):
            grad = grad.sum(axis=0)
        for i, dim in enumerate(ts2.shape):
            if dim == 1:
                grad = grad.sum(axis=i, keepdims=True)
        return grad

    return build_binary_ops_tensor(
        ts1, ts2, grad_fn_ts1, grad_fn_ts2, values)


def dot_(ts1, ts2):
    #values = ts1.values @ ts2.values
    from core.tensor import CTX, QUEUE
    shape = tuple(list(ts1.shape)[:-1] + list(ts2.shape)[1:])
    values = cl_array.empty(QUEUE, shape, dtype=np.float32)
    # TODO: make it faster (https://cnugteren.github.io/tutorial/pages/page3.html)
    op = cl_build("dot_op", """
    __kernel void dot_op(const int N, const int K,
        __global const float *A, __global const float *B, __global float *C) {
      const int gRow = get_global_id(0);  // range 0..M
      const int gCol = get_global_id(1);  // range 0..N
      float acc = 0.0f;
      for (int k=0; k<K; k++) {
        acc += A[gRow*K + k] * B[k*N + gCol];
      }
      C[gRow*N + gCol] = acc;
    }
    """)
    N, K = np.int32(ts2.shape[-1]), np.int32(ts1.shape[-1])
    op(shape, None, N, K, ts1.values.data, ts2.values.data, values.data)

    # c = a @ b
    # D_c / D_a = grad @ b.T
    # D_c / D_b = a.T @ grad
    def grad_fn_ts1(grad):
        return grad @ ts2.values.T

    def grad_fn_ts2(grad):
        return ts1.values.T @ grad

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


def max_(ts, axis=None):
    values = np.max(ts.values, axis=axis)

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


def sum_(ts, axis):
    values = ts.values.sum(axis=axis)
    if axis is not None:
        repeat = ts.values.shape[axis]

    def grad_fn(grad):
        if axis is None:
            grad = grad * np.ones_like(ts.values)
        else:
            grad = np.expand_dims(grad, axis)
            grad = np.repeat(grad, repeat, axis)
        return grad

    return build_unary_ops_tensor(ts, grad_fn, values)


def transpose_(ts, axes=None):
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
    #values = unary_op("neg", ts.values, ts.shape)
    values = -ts.values

    def grad_fn(grad):
        grad_values = unary_op("neg", grad.values, grad.shape)
        grad = grad.__class__(grad_values, shape=grad.shape, gpu=True)
        return grad

    return build_unary_ops_tensor(ts, grad_fn, values)


def reshape_(ts, newshape):
    shape = ts.values.shape
    values = ts.values.reshape(newshape)

    def grad_fn(grad):
        return grad.reshape(shape)

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


def max(obj, axis=None):
    return max_(as_tensor(obj), axis=axis)


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

