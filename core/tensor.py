import numpy as np
import core.autograd.ops as ops
from env import GRAPH, BACKEND
from core.dtype import float32

from core.backend.numpy import NpArray as CPUArray
GPUArray = type(None)
if BACKEND == "opencl":
    from core.backend.opencl import ClArray as GPUArray
elif BACKEND == "cuda":
    from core.backend.cuda import CuArray as GPUArray

def as_tensor(obj):
    if not isinstance(obj, Tensor):
        obj = Tensor(obj)
    return obj

class Tensor:
    def __init__(self, values, requires_grad=False, dependency=(), dtype=float32, name=None):
        self._gpu = isinstance(values, GPUArray)
        self.array = values if isinstance(values, (CPUArray, GPUArray)) else CPUArray(values, dtype=dtype)
        self.dtype = dtype

        self.grad = None
        self.requires_grad = requires_grad
        self.dependency = dependency

        self.name = name
        self.outdegree = 0
        self.bwdcost = 0

    def to(self, device):
        assert device in ("cpu", "gpu"), f"Device {device} not support yet."
        return getattr(self, device)()

    def gpu(self):
        assert GPUArray != type(None), f"backend {BACKEND} not support gpu device"
        if not self._gpu:
            return Tensor(values=GPUArray(self.array.numpy()),
                          requires_grad=self.requires_grad,
                          dependency=self.dependency,
                          dtype=self.dtype,
                          name=self.name)
        return self

    def cpu(self):
        if self._gpu:
            return Tensor(values=CPUArray(self.array.numpy()),
                          requires_grad=self.requires_grad,
                          dependency=self.dependency,
                          dtype=self.dtype,
                          name=self.name)
        return self

    def numpy(self):
        return self.array.numpy()

    @property
    def values(self):
        return self.array

    @values.setter
    def values(self, new_values):
        self.array = new_values
        self.grad = None

    @property
    def shape(self):
        return self.array.shape

    def __repr__(self):
        return (f"Tensor(name={self.name}, shape={self.shape}, "
                f"requires_grad={self.requires_grad}, "
                f"gpu={self._gpu}, array={self.array.__class__.__name__})")

    def __gt__(self, other):
        return ops.gt(self, as_tensor(other))

    def __eq__(self, other):
        return ops.eq(self, as_tensor(other))

    def __ge__(self, other):
        return ops.ge(self, as_tensor(other))

    def __add__(self, other):
        return ops.add(self, as_tensor(other))

    def __radd__(self, other):
        return ops.add(as_tensor(other), self)

    def __iadd__(self, other):
        self.values += as_tensor(other).values
        return self

    def __sub__(self, other):
        return ops.sub(self, as_tensor(other))

    def __rsub__(self, other):
        return ops.sub(as_tensor(other), self)

    def __isub__(self, other):
        self.array = self.array - as_tensor(other).values
        return self

    def __mul__(self, other):
        return ops.mul(self, as_tensor(other))

    def __rmul__(self, other):
        return ops.mul(as_tensor(other), self)

    def __imul__(self, other):
        self.values *= as_tensor(other).values
        return self

    def __truediv__(self, other):
        return ops.div(self, as_tensor(other))

    def __rtruediv__(self, other):
        return ops.div(as_tensor(other), self)

    def __itruediv__(self, other):
        self.values = self.values / as_tensor(other).values
        return self

    def __neg__(self):
        return ops.neg(self)

    def __getitem__(self, key):
        return ops.getitem(self, key)

    def __pow__(self, other):
        return ops.pow(self, as_tensor(other))

    def __rpow__(self, other):
        return ops.pow(as_tensor(other), self)

    def __ipow__(self, other):
        self.values = self.values ** as_tensor(other).values
        return self

    def __matmul__(self, other):
        return ops.matmul(self, as_tensor(other))

    def __rmatmul__(self, other):
        return ops.matmul(as_tensor(other), self)

    def __imatmul__(self, other):
        self.values = self.values @ as_tensor(other).values
        return self

    def __len__(self):
        assert self.shape, "Error getting length of a 0-d tensor"
        return self.shape[0]

    def sum(self, axis=None, keepdims=False):
        return ops.sum(self, axis=axis, keepdims=keepdims)

    def max(self, axis=None, keepdims=False):
        return ops.max(self, axis=axis, keepdims=keepdims)

    def min(self, axis=None, keepfims=False):
        return ops.min(self, axis=axis, keepdims=keepdims)

    def permute(self, axes=None):
        return ops.permute(self, axes=axes)

    def log(self):
        return ops.log(self)

    def reshape(self, newshape):
        return ops.reshape(self, newshape)

    def flatten(self):
        return ops.flatten(self)

    def relu(self):
        return ops.relu(self)

    def exp(self):
        return ops.exp(self)

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def T(self):
        return ops.permute(self, axes=None)

    def backward(self, grad=None):
        assert self.requires_grad, "Call backward() on a non-requires-grad tensor."
        self.outdegree -= 1
        if grad is None:
            grad = GPUArray([1.0]) if self._gpu else CPUArray([1.0])
            self.outdegree = 0
        if self._gpu and not isinstance(grad, GPUArray):
            grad = GPUArray(grad, dtype=self.dtype)
        if not self._gpu and not isinstance(grad, CPUArray):
            grad = CPUArray(grad, dtype=self.dtype)

        if self.requires_grad:
            self.grad = grad if self.grad is None else self.grad + grad

        if self.outdegree <= 0:
            for dep in self.dependency:
                grad_for_dep = dep["grad_fn"](self.grad)
                if GRAPH:
                    grad_for_dep, cost = grad_for_dep
                    self.bwdcost += cost
                dep["tensor"].backward(grad_for_dep)

    def zero_grad(self):
        self.grad = None
