"""Tensor wraps numpy ndarray with some stuffs for pytorch-like autograd."""

import numpy as np
import core.ops as ops
from core.ndarray import GPUArray

import time
import os
DEBUG = int(os.getenv("DEBUG", "0"))
OPT = int(os.getenv("OPT", "0"))

def as_tensor(obj):
    if not isinstance(obj, Tensor):
        obj = Tensor(obj)
    return obj


class Tensor:

    def __init__(self,
                 values,
                 requires_grad=False,
                 dependency=(),
                 dtype=np.float32,
                 name=None):
        self._gpu = isinstance(values, GPUArray)
        self.values = values if self._gpu else np.asarray(values, dtype)
        self.dtype = dtype

        self.name = name
        self.outdegree = 0
        self.bwdcost = 0

        self.grad = None
        self.requires_grad = requires_grad
        self.dependency = dependency

    def gpu(self):
        if not self._gpu:
            return Tensor(values=GPUArray(self._values),
                          requires_grad=self.requires_grad,
                          dependency=self.dependency,
                          dtype=self.dtype,
                          name=self.name)
        return self

    def cpu(self):
        if self._gpu:
            return Tensor(values=self._values.numpy(),
                          requires_grad=self.requires_grad,
                          dependency=self.dependency,
                          dtype=self.dtype,
                          name=self.name)
        return self

    def numpy(self):
        if self._gpu:
            return self._values.numpy()
        return self._values

    @property
    def values(self):
        return self._values

    @values.setter
    def values(self, new_values):
        self._values = new_values
        #self.grad = None

    @property
    def shape(self):
        return self._values.shape

    def __repr__(self):
        return (f"Tensor(name={self.name}, shape={self.shape}, "
                f"requires_grad={self.requires_grad}, "
                f"gpu={self._gpu})")

    def __gt__(self, other):
        return self.values > as_tensor(other).values

    def __ge__(self, other):
        return self.values >= as_tensor(other).values

    def __eq__(self, other):
        return self.values == as_tensor(other).values

    # TODO: programmatically register
    def __add__(self, other):
        return ops.add_(self, as_tensor(other))

    def __radd__(self, other):
        return ops.add_(as_tensor(other), self)

    def __iadd__(self, other):
        self.values += as_tensor(other).values
        return self

    def __sub__(self, other):
        return ops.sub_(self, as_tensor(other))

    def __rsub__(self, other):
        return ops.sub_(as_tensor(other), self)

    def __isub__(self, other):
        self._values = self._values - as_tensor(other).values
        return self

    def __mul__(self, other):
        return ops.mul_(self, as_tensor(other))

    def __rmul__(self, other):
        return ops.mul_(as_tensor(other), self)

    def __imul__(self, other):
        self.values *= as_tensor(other).values
        return self

    def __truediv__(self, other):
        return ops.div_(self, as_tensor(other))

    def __rtruediv__(self, other):
        return ops.div_(as_tensor(other), self)

    def __itruediv__(self, other):
        self.values = self.values / as_tensor(other).values
        return self

    def __neg__(self):
        return ops.neg_(self)

    def __getitem__(self, key):
        return ops.getitem_(self, key)

    def __pow__(self, other):
        return ops.pow_(self, as_tensor(other))

    def __rpow__(self, other):
        return ops.pow_(as_tensor(other), self)

    def __ipow__(self, other):
        self.values = self.values ** as_tensor(other).values
        return self

    def __matmul__(self, other):
        return ops.matmul_(self, as_tensor(other))

    def __rmatmul__(self, other):
        return ops.matmul_(as_tensor(other), self)

    def __imatmul__(self, other):
        self.values = self.values @ as_tensor(other).values
        return self

    def __len__(self):
        assert self.shape, "Error getting length of a 0-d tensor"
        return self.shape[0]

    def sum(self, axis=None, keepdims=False):
        return ops.sum_(self, axis=axis, keepdims=keepdims)

    def max(self, axis=None, keepdims=False):
        return ops.max_(self, axis=axis, keepdims=keepdims)

    def min(self, axis=None):
        return ops.min_(self, axis=axis)

    def transpose(self, axes=None):
        return ops.transpose_(self, axes=axes)

    def log(self):
        return ops.log_(self)

    def reshape(self, newshape):
        return ops.reshape_(self, newshape)

    def flatten(self):
        return ops.flatten_(self)

    def clip(self, min_=None, max_=None):
        return ops.clip_(self, min_, max_)

    def relu(self, inplace=False):
        return ops.relu_(self, inplace)

    def exp(self):
        return ops.exp_(self)

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def T(self):
        return ops.transpose_(self, axes=None)

    def backward(self, grad=None):
        assert self.requires_grad, "Call backward() on a non-requires-grad tensor."
        self.outdegree -= 1
        if grad is None:
            grad = GPUArray(1.0) if self._gpu else np.array(1.0, dtype=np.float32)
            self.outdegree = 0

        if OPT:
            #TODO: raise error on Nvidia device
            if self.requires_grad and self.grad is None:
                self.grad = grad
            else:
                self.grad = self.grad + grad
        else:
            if self.requires_grad and self.grad is None:
                self.zero_grad()
            self.grad = self.grad + grad

        if not self.outdegree:
            for dep in self.dependency:
                #grad_for_dep, cost = dep["grad_fn"](self.grad)
                #self.bwdcost += cost
                grad_for_dep = dep["grad_fn"](self.grad)
                dep["tensor"].backward(grad_for_dep)

    def zero_grad(self):
        if self.grad is None:
            self.grad = GPUArray(0.0).reshape([1]*self.ndim).expand(self.shape)
        else:
            self.grad.fill(0.0)
