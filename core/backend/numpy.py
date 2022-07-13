import numpy as np

from core.backend.base import Array
from utils.dtype import float32

class NpArray(Array):
    def __init__(self, data, shape=None, dtype=float32):
        super().__init__(shape, dtype)
        self.data = np.asarray(data)
        self.shape = self.data.shape

    @property
    def size(self): return self.data.nbytes
    @property
    def ndim(self): return len(self.data.shape)

    # ##### Unary Ops #####
    def neg(self): return self.asarray(np.negative(self.data))
    def exp(self): return self.asarray(np.exp(self.data))
    def log(self): return self.asarray(np.log(self.data))
    def relu(self): return self.asarray(np.maximum(self.data, 0))

    # ##### Binary Ops #####
    def add(self, other, out=None): return self.asarray(self.data + other.data)
    def sub(self, other, out=None): return self.asarray(self.data - other.data)
    def div(self, other, out=None): return self.asarray(self.data / other.data)
    def mul(self, other, out=None): return self.asarray(self.data * other.data)
    def pow(self, other, out=None): return self.asarray(self.data ** other.data)
    def eq(self, other): return self.asarray(self.data == other.data)
    def ge(self, other): return self.asarray(self.data >= other.data)
    def gt(self, other): return self.asarray(self.data > other.data)
    def le(self, other): return self.asarray(self.data <= other.data)
    def lt(self, other): return self.asarray(self.data < other.data)
    def matmul(self, other): return self.asarray(self.data @ other.data)
    def drelu(self, other): return self.asarray((self.data > 0) * other.data)

    # ##### Reduce Ops #####
    def sum(self, axis=None, keepdims=False): return self.asarray(np.sum(self.data, axis=axis, keepdims=keepdims))
    def max(self, axis=None, keepdims=False): return self.asarray(np.max(self.data, axis=axis, keepdims=keepdims))

    # ##### Slice Ops #####
    def __getitem__(self, key): return self.asarray(self.data[key])
    def __setitem__(self, key, value): self.data[key] = value.data

    # ##### Movement Ops #####
    def reshape(self, shape): return self.asarray(np.reshape(self.data, shape))
    def expand(self, shape): return self.asarray(np.broadcast_to(self.data, shape))
    def squeeze(self, axis=None): return self.asarray(np.squeeze(self.data, shape))
    def permute(self, axes): return self.asarray(np.transpose(self.data, axes))

    # ##### Construct Ops #####
    @classmethod
    def empty(cls, shape, dtype=float32): return cls.asarray(np.empty(shape, dtype))
    @classmethod
    def zeros(cls, shape, dtype=float32): return cls.asarray(np.zeros(shape, dtype))
    @classmethod
    def ones(cls, shape, dtype=float32): return cls.asarray(np.ones(shape, dtype))
    @classmethod
    def full(cls, shape, value, dtype=float32): return cls.asarray(np.full(shape, value, dtype))
    @classmethod
    def uniform(cls, a, b, shape, dtype=float32):
        return cls.asarray(np.random.uniform(a, b, shape).astype(dtype))
    @classmethod
    def normal(cls, loc, scale, shape, dtype=float32):
        return cls.asarray(np.random.normal(loc, scale, shape).astype(dtype))

    def numpy(self):
        return self.data.copy()

