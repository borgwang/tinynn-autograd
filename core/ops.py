import numpy as np

def genname(prefix, *args):
    return f"{prefix}_" + "_".join(str(id(ts))[-4:] for ts in args)

def as_tensor(obj):
    from core.tensor import as_tensor
    return as_tensor(obj)

def build_binary_ops_tensor(ts1, ts2, grad_fn_ts1, grad_fn_ts2, values, name):
    requires_grad = ts1.requires_grad or ts2.requires_grad
    dependency = []
    if ts1.requires_grad:
        dependency.append(dict(tensor=ts1, grad_fn=grad_fn_ts1))
    if ts2.requires_grad:
        dependency.append(dict(tensor=ts2, grad_fn=grad_fn_ts2))
    tensor_cls = ts1.__class__
    return tensor_cls(values, requires_grad, dependency, name=name)

def build_unary_ops_tensor(ts, grad_fn, values, name):
    requires_grad = ts.requires_grad
    dependency = []
    if ts.requires_grad:
        dependency.append(dict(tensor=ts, grad_fn=grad_fn))
    tensor_cls = ts.__class__
    return tensor_cls(values, requires_grad, dependency, name=name)

def add_(ts1, ts2):
    values = ts1.values + ts2.values
    def grad_fn_ts1(grad):
        for _ in range(grad.ndim - ts1.values.ndim):
            grad = grad.sum(axis=0)
        for i, dim in enumerate(ts1.shape):
            if dim == 1:
                grad = grad.sum(axis=i, keepdims=True)
        return grad
    def grad_fn_ts2(grad):
        for _ in range(grad.ndim - ts2.values.ndim):
            grad = grad.sum(axis=0)
        for i, dim in enumerate(ts2.shape):
            if dim == 1:
                grad = grad.sum(axis=i, keepdims=True)
        return grad
    name = genname("add", ts1, ts2)
    return build_binary_ops_tensor(ts1, ts2, grad_fn_ts1, grad_fn_ts2, values, name=name)

def sub_(ts1, ts2):
    values = ts1.values - ts2.values
    def grad_fn_ts1(grad):
        for _ in range(grad.ndim - ts1.values.ndim):
            grad = grad.sum(axis=0)
        for i, dim in enumerate(ts1.shape):
            if dim == 1:
                grad = grad.sum(axis=i, keepdims=True)
        return grad
    def grad_fn_ts2(grad):
        for _ in range(grad.ndim - ts2.values.ndim):
            grad = grad.sum(axis=0)
        for i, dim in enumerate(ts2.shape):
            if dim == 1:
                grad = grad.sum(axis=i, keepdims=True)
        return -grad
    name = genname("sub", ts1, ts2)
    return build_binary_ops_tensor(ts1, ts2, grad_fn_ts1, grad_fn_ts2, values, name=name)

def mul_(ts1, ts2):
    values = ts1.values * ts2.values
    grad_fn_ts1 = lambda g: g * ts2.values
    grad_fn_ts2 = lambda g: g * ts1.values
    name = genname("mul", ts1, ts2)
    return build_binary_ops_tensor(ts1, ts2, grad_fn_ts1, grad_fn_ts2, values, name=name)

def div_(ts1, ts2):
    values = ts1.values / ts2.values
    def grad_fn_ts1(grad):
        grad = grad / ts2.values
        for _ in range(grad.ndim - ts1.values.ndim):
            grad = grad.sum(axis=0)
        for i, dim in enumerate(ts1.shape):
            if dim == 1:
                grad = grad.sum(axis=i, keepdims=True)
        return grad

    def grad_fn_ts2(grad):
        grad = -grad * ts1.values / (ts2.values * ts2.values)
        for _ in range(grad.ndim - ts2.values.ndim):
            grad = grad.sum(axis=0)
        for i, dim in enumerate(ts2.shape):
            if dim == 1:
                grad = grad.sum(axis=i, keepdims=True)
        return grad

    name = genname("div", ts1, ts2)
    return build_binary_ops_tensor(ts1, ts2, grad_fn_ts1, grad_fn_ts2, values, name=name)

def pow_(ts1, ts2):
    values = ts1.values ** ts2.values
    def grad_fn_ts1(grad):
        return grad * (ts2.values * ts1.values ** (ts2.values - np.ones((), dtype=np.float32)))
    def grad_fn_ts2(grad):
        grad = grad * (np.log(ts1.values) * values)
        for _ in range(grad.ndim - ts2.values.ndim):
            grad = grad.sum(axis=0)
        for i, dim in enumerate(ts2.shape):
            if dim == 1:
                grad = grad.sum(axis=i, keepdims=True)
        return grad
    name = genname("pow", ts1, ts2)
    return build_binary_ops_tensor(ts1, ts2, grad_fn_ts1, grad_fn_ts2, values, name=name)

def matmul_(ts1, ts2):
    values = ts1.values @ ts2.values
    def grad_fn_ts1(grad):
        if grad.shape == (128, 1):
            import pdb; pdb.set_trace()
        return grad @ ts2.values.T
    #grad_fn_ts1 = lambda g: g @ ts2.values.T
    grad_fn_ts2 = lambda g: ts1.values.T @ g
    name = genname("matmul", ts1, ts2)
    return build_binary_ops_tensor(ts1, ts2, grad_fn_ts1, grad_fn_ts2, values, name=name)


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
    return build_binary_ops_tensor(ts1, ts2, grad_fn_ts1, grad_fn_ts2, values)


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
    return build_binary_ops_tensor(ts1, ts2, grad_fn_ts1, grad_fn_ts2, values)

def exp_(ts):
    values = ts.values.exp()
    def grad_fn(grad):
        return values * grad
    name = genname("exp", ts)
    return build_unary_ops_tensor(ts, grad_fn, values, name=name)

def max_(ts, axis, keepdims):
    values = ts.values.max(axis=axis, keepdims=keepdims)
    def grad_fn(grad):
        return grad * (values == ts.values)
    name = genname("max", ts)
    return build_unary_ops_tensor(ts, grad_fn, values, name=name)

def min_(ts, axis=None):
    values = np.min(ts.values, axis=axis)
    def grad_fn(grad):
        return grad * (ts.values.min(axis=axis, keepdims=1) == ts.values)
    name = genname("min", ts)
    return build_unary_ops_tensor(ts, grad_fn, values, name=name)

def log_(ts):
    values = ts.values.log()
    def grad_fn(grad):
        return grad / ts.values
    name = genname("log", ts)
    return build_unary_ops_tensor(ts, grad_fn, values, name=name)

def sum_(ts, axis, keepdims):
    values = ts.values.sum(axis, keepdims)
    def grad_fn(grad):
        if axis is None:
            return grad * grad.__class__.ones(ts.shape)
        else:
            if not keepdims:
                grad = grad.reshape((*ts.shape[:axis],1,*ts.shape[axis+1:]))
            return grad.expand(ts.shape)
    name = genname("sum", ts)
    return build_unary_ops_tensor(ts, grad_fn, values, name=name)

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

def relu_(ts, inplace):
    values = ts.values.relu(inplace=inplace)
    def grad_fn(grad):
        return grad * (ts.values > 0.0)
    name = genname("relu", ts)
    return build_unary_ops_tensor(ts, grad_fn, values, name=name)

def getitem_(ts, key):
    values = ts.values[key]
    def grad_fn(grad):
        recover_grad = np.zeros_like(ts.values)
        recover_grad[key] = grad
        return recover_grad
    name = genname("getitem", ts)
    return build_unary_ops_tensor(ts, grad_fn, values, name=name)

def neg_(ts):
    values = -ts.values
    def grad_fn(grad):
        return -grad
    name = genname("neg", ts)
    return build_unary_ops_tensor(ts, grad_fn, values, name=name)

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

def maximum(obj1, obj2):
    return maximum_(as_tensor(obj1), as_tensor(obj2))

def minimum(obj1, obj2):
    return minimum_(as_tensor(obj1), as_tensor(obj2))

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
