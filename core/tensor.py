"""Tensor wraps numpy ndarray with some stuffs for pytorch-like autograd."""

import os

import numpy as np
#import core.ops as ops
import core.gpu_ops as ops
from core.ndarray import GPUArray
import pyopencl as cl


def as_tensor(obj):
    if not isinstance(obj, Tensor):
        obj = Tensor(obj)
    return obj


class Tensor:

    def __init__(self,
                 values,
                 requires_grad=False,
                 dependency=None,
                 dtype=np.float32):
        gpu = isinstance(values, GPUArray)
        self.values = values if gpu else np.asarray(values, dtype)
        self._shape = self._values.shape
        self._gpu = gpu

        self.grad = None
        self.requires_grad = requires_grad
        #if self.requires_grad:
        #    self.zero_grad()
        self.dependency = dependency
        if self.dependency is None:
            self.dependency = []

    def gpu(self):
        if not self._gpu:
            return Tensor(values=GPUArray(self._values),
                          requires_grad=self.requires_grad,
                          dependency=self.dependency,
                          dtype=self._values.dtype)
        return self

    def cpu(self):
        if self._gpu:
            return self._values.to_cpu()
        return self._values

    @property
    def values(self):
        return self._values

    @values.setter
    def values(self, new_values):
        #self._values = np.asarray(new_values)
        self._values = new_values
        self.grad = None

    @property
    def shape(self):
        return self._values.shape

    def __repr__(self):
        return (f"Tensor(shape={self.shape}, "
                f"requires_grad={self.requires_grad}, "
                f"gpu={self._gpu})")

    def __gt__(self, other):
        return self.values > as_tensor(other).values

    def __lt__(self, other):
        return self.values < as_tensor(other).values

    def __ge__(self, other):
        return self.values >= as_tensor(other).values

    def __le__(self, other):
        return self.values <= as_tensor(other).values

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
        self.values = self.values - as_tensor(other).values
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
        return len(self.values)

    def sum(self, axis=None):
        return ops.sum_(self, axis=axis)

    def max(self, axis=None):
        return ops.max_(self, axis=axis)

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

    @property
    def T(self):
        return ops.transpose_(self, axes=None)

    def backward(self, grad=None):
        assert self.requires_grad, "Call backward() on a non-requires-grad tensor."
        if grad is None:
            if self._gpu:
                grad = GPUAarray.ones(self.shape, dtype=np.float32)
            else:
                grad = np.ones(self.shape, dtype=np.float32)

        # accumulate the gradients
        self.grad += grad
        # propagate the gradients to its dependencies
        for dep in self.dependency:
            grad_for_dep = dep["grad_fn"](grad)
            dep["tensor"].backward(grad_for_dep)

    def zero_grad(self):
        if self.grad is None:
            if self._gpu:
                self.grad = GPUArray.zeros(self.shape, dtype=np.float32)
            else:
                self.grad = np.zeros(self.shape, dtype=np.float32)
        else:
            self.grad *= 0.0
