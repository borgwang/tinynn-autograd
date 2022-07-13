import numpy as np
from utils.helper import timer, genname
from utils.math import argsort
from env import GRAPH

def as_tensor(obj):
    from core.tensor import as_tensor
    return as_tensor(obj)

def binary_ops(func):
    def wrapper(*args, **kwargs):
        ts1, ts2, grad_fn1, grad_fn2, values = func(*args, **kwargs)
        requires_grad = (ts1.requires_grad and grad_fn1) or (ts2.requires_grad and grad_fn2)
        dependency = []
        if ts1.requires_grad and grad_fn1:
            if GRAPH: grad_fn1=timer(grad_fn1)
            dependency.append(dict(tensor=ts1, grad_fn=grad_fn1))
            ts1.outdegree += 1
        if ts2.requires_grad and grad_fn2:
            if GRAPH: grad_fn2=timer(grad_fn2)
            dependency.append(dict(tensor=ts2, grad_fn=grad_fn2))
            ts2.outdegree += 1
        name = genname(func.__name__, ts1, ts2)
        return ts1.__class__(values, requires_grad, dependency, name=name)
    return wrapper

def unary_ops(func):
    def wrapper(*args, **kwargs):
        ts, grad_fn, values = func(*args, **kwargs)
        requires_grad = ts.requires_grad and grad_fn
        dependency = []
        if ts.requires_grad and grad_fn:
            if GRAPH: grad_fn=timer(grad_fn)
            dependency.append(dict(tensor=ts, grad_fn=grad_fn))
            ts.outdegree += 1
        name = genname(func.__name__, ts)
        return ts.__class__(values, requires_grad, dependency, name=name)
    return wrapper

@binary_ops
def add_(ts1, ts2):
    values = ts1.values + ts2.values
    def grad_fn1(grad):
        for _ in range(grad.ndim - ts1.values.ndim):
            grad = grad.sum(axis=0)
        for i, dim in enumerate(ts1.shape):
            if dim == 1:
                grad = grad.sum(axis=i, keepdims=True)
        return grad
    def grad_fn2(grad):
        for _ in range(grad.ndim - ts2.values.ndim):
            grad = grad.sum(axis=0)
        for i, dim in enumerate(ts2.shape):
            if dim == 1:
                grad = grad.sum(axis=i, keepdims=True)
        return grad
    return ts1, ts2, grad_fn1, grad_fn2, values

@binary_ops
def sub_(ts1, ts2):
    values = ts1.values - ts2.values
    def grad_fn1(grad):
        for _ in range(grad.ndim - ts1.values.ndim):
            grad = grad.sum(axis=0)
        for i, dim in enumerate(ts1.shape):
            if dim == 1:
                grad = grad.sum(axis=i, keepdims=True)
        return grad
    def grad_fn2(grad):
        for _ in range(grad.ndim - ts2.values.ndim):
            grad = grad.sum(axis=0)
        for i, dim in enumerate(ts2.shape):
            if dim == 1:
                grad = grad.sum(axis=i, keepdims=True)
        return -grad
    return ts1, ts2, grad_fn1, grad_fn2, values

@binary_ops
def mul_(ts1, ts2):
    values = ts1.values * ts2.values
    def grad_fn1(grad):
        return ts2.values * grad
    def grad_fn2(grad):
        return ts1.values * grad
    return ts1, ts2, grad_fn1, grad_fn2, values

@binary_ops
def div_(ts1, ts2):
    values = ts1.values / ts2.values
    def grad_fn1(grad):
        grad = grad / ts2.values
        for _ in range(grad.ndim - ts1.values.ndim):
            grad = grad.sum(axis=0)
        for i, dim in enumerate(ts1.shape):
            if dim == 1:
                grad = grad.sum(axis=i, keepdims=True)
        return grad
    def grad_fn2(grad):
        grad = -grad * values / ts2.values
        for _ in range(grad.ndim - ts2.values.ndim):
            grad = grad.sum(axis=0)
        for i, dim in enumerate(ts2.shape):
            if dim == 1:
                grad = grad.sum(axis=i, keepdims=True)
        return grad
    return ts1, ts2, grad_fn1, grad_fn2, values

@binary_ops
def pow_(ts1, ts2):
    values = ts1.values ** ts2.values
    def grad_fn1(grad):
        return grad * (ts2.values * ts1.values ** (ts2.values - 1.0))
    def grad_fn2(grad):
        grad = grad * (values * ts1.values.log())
        for _ in range(grad.ndim - ts2.values.ndim):
            grad = grad.sum(axis=0)
        for i, dim in enumerate(ts2.shape):
            if dim == 1:
                grad = grad.sum(axis=i, keepdims=True)
        return grad
    return ts1, ts2, grad_fn1, grad_fn2, values

@binary_ops
def matmul_(ts1, ts2):
    values = ts1.values @ ts2.values
    def grad_fn1(grad):
        return grad @ ts2.values.T
    def grad_fn2(grad):
        return ts1.values.T @ grad
    return ts1, ts2, grad_fn1, grad_fn2, values

@binary_ops
def gt_(ts1, ts2):
    values = ts1.values > ts2.values
    return ts1, ts2, None, None, values

@binary_ops
def eq_(ts1, ts2):
    values = ts1.values == ts2.values
    return ts1, ts2, None, None, values

@binary_ops
def ge_(ts1, ts2):
    values = ts1.values >= ts2.values
    return ts1, ts2, None, None, values

@unary_ops
def exp_(ts):
    values = ts.values.exp()
    def grad_fn(grad):
        return values * grad
    return ts, grad_fn, values

@unary_ops
def max_(ts, axis, keepdims):
    values = ts.values.max(axis=axis, keepdims=keepdims)
    def grad_fn(grad):
        return grad * (values == ts.values)
    return ts, grad_fn, values

@unary_ops
def min_(ts, axis, keepdims):
    values = ts.values.min(axis=axis, keepdims=keepdims)
    def grad_fn(grad):
        return grad * (values == ts.values)
    return ts, grad_fn, values

@unary_ops
def log_(ts):
    values = ts.values.log()
    def grad_fn(grad):
        return grad / ts.values
    return ts, grad_fn, values

@unary_ops
def sum_(ts, axis, keepdims):
    values = ts.values.sum(axis=axis, keepdims=keepdims)
    def grad_fn(grad):
        if axis is None:
            return grad.reshape([1] * ts.ndim).expand(ts.shape)
        else:
            if not keepdims:
                grad = grad.reshape((*ts.shape[:axis],1,*ts.shape[axis+1:]))
            return grad.expand(ts.shape)
    return ts, grad_fn, values

@unary_ops
def relu_(ts):
    values = ts.values.relu()
    def grad_fn(grad):
        return grad.drelu(ts.values)
    return ts, grad_fn, values

@unary_ops
def neg_(ts):
    values = -ts.values
    def grad_fn(grad):
        return -grad
    return ts, grad_fn, values

@unary_ops
def reshape_(ts, newshape):
    oldshape = ts.values.shape
    values = ts.values.reshape(newshape)
    def grad_fn(grad):
        return grad.reshape(oldshape)
    return ts, grad_fn, values

@unary_ops
def permute_(ts, axes=None):
    if axes is None:
        axes = range(ts.values.ndim)[::-1]
    axes = list(axes)
    values = ts.values.permute(axes)
    def grad_fn(grad):
        return grad.permute(argsort(axes))
    return ts, grad_fn, values

@unary_ops
def getitem_(ts, key):
    values = ts.values[key]
    def grad_fn(grad):
        ret = grad.__class__.zeros(ts.shape)
        ret[key] = grad
        return ret
    return ts, grad_fn, values

# TODO: implement ops below
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


