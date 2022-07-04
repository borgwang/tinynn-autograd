import copy

import numpy as np
import pyopencl as cl

from core.ops_gpu import cl_ctx, cl_queue, cl_rng, alloc_buffer
from core.ops_gpu import binary_op, matmul_op, unary_op, contiguous_op, reduce_op
from utils.math import prod


def as_gpu_array(obj):
    if not isinstance(obj, GPUArray):
        obj = GPUArray(obj)
    return obj


class GPUArray:

    def __init__(self, data=None, shape=None, dtype=np.float32, buffer=None):
        if data is not None:
            data = np.asarray(data, dtype=dtype)
        shape = tuple(shape) if shape is not None else tuple(data.shape)
        self.buffer = alloc_buffer(shape, dtype, data) if buffer is None else buffer

        self.strides = tuple(prod(shape[i+1:]) for i in range(len(shape)))
        self.shape, self.dtype = shape, dtype
        self.__c_contiguous, self.__f_contiguous = True, False
        self.__update_contiguousness()
        self.register_ops()

    def __repr__(self):
        return (f"<GPUArray dtype={self.dtype} shape={self.shape} strides={self.strides} size={self.size} contiguous=({int(self.__c_contiguous)}, {int(self.__f_contiguous)})>")

    def register_ops(self):
        cls = self.__class__
        for op in ("add", "sub", "mul", "truediv", "pow"):
            setattr(cls, f"__{op}__",
                    (lambda op: lambda a, b: binary_op(op, a, as_gpu_array(b)))(op))
            setattr(cls, f"__i{op}__",
                    (lambda op: lambda a, b: binary_op(op, a, as_gpu_array(b), ret=a))(op))
            setattr(cls, f"__r{op}__",
                    (lambda op: lambda a, b: binary_op(op, as_gpu_array(b), a))(op))
        setattr(cls, f"__matmul__", lambda a, b: matmul_op(a, as_gpu_array(b)))
        setattr(cls, f"__neg__", lambda a: unary_op("neg", a))

    @property
    def size(self):
        return self.buffer.size

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def c_contiguous(self):
        return self.__c_contiguous

    @property
    def f_contiguous(self):
        return self.__f_contiguous

    @classmethod
    def empty(cls, shape, dtype=np.float32):
        return cls(shape=shape, dtype=dtype)

    @classmethod
    def zeros(cls, shape, dtype=np.float32):
        return cls(shape=shape, dtype=dtype).fill(0)

    @classmethod
    def ones(cls, shape, dtype=np.float32):
        return cls(shape=shape, dtype=dtype).fill(1)

    @classmethod
    def full(cls, shape, value, dtype=np.float32):
        return cls(shape=shape, dtype=dtype).fill(value)

    @classmethod
    def from_numpy(cls, arr):
        return cls(data=arr)

    @classmethod
    def uniform(cls, a, b, shape, dtype):
        buffer = cl_rng.uniform(a=a, b=b, shape=shape, dtype=dtype, cq=cl_queue).data  # cheating
        return cls(shape=shape, dtype=dtype, buffer=buffer)

    @classmethod
    def normal(cls):
        pass

    def numpy(self):
        data = np.empty(self.shape, dtype=self.dtype)
        cl.enqueue_copy(cl_queue, data, self.contiguous().buffer, is_blocking=True)
        return data

    def contiguous(self):
        return contiguous_op(self)

    def reshape(self, shape):
        if -1 in shape:
            size = prod(self.shape)
            assert shape.count(-1) <= 1, "Only one dimension can be inferred"
            axis = shape.index(-1)
            infer = prod([s for s in shape if s != -1])
            assert size % infer == 0, f"Shape {shape} invalid for size {size}"
            shape = (*shape[:axis], size // infer, *shape[axis+1:])

        assert prod(shape) == prod(self.shape), f"Can not reshape {self.shape} to {shape}"
        if self.__c_contiguous or self.__f_contiguous:
            inst = copy.copy(self)
            if self.__c_contiguous:
                strides = (prod(shape[i+1:]) for i in range(len(shape)))
            else:
                strides = (prod(shape[:i]) for i in range(len(shape)))
            inst.shape, inst.strides = tuple(shape), tuple(strides)
            inst.__update_contiguousness()
        else:
            inst = self.contiguous().reshape(shape)
        return inst

    def expand(self, shape):
        inst = copy.copy(self)
        assert len(shape) == inst.ndim
        strides = []
        for i, (s1, s2) in enumerate(zip(inst.shape, shape)):
            if s1 < s2:
                assert s1 == 1
            strides.append(0 if s1 < s2 else inst.strides[i])
        inst.shape, inst.strides = tuple(shape), tuple(strides)
        inst.__update_contiguousness()
        return inst

    def squeeze(self, axis=None):
        if axis is None:
            axis = [i for i, s in enumerate(self.shape) if s == 1]
        elif isinstance(axis, int):
            axis = [axis]
        assert isinstance(axis, (list, tuple))
        axis = [a if a != -1 else self.ndim - 1 for a in axis]
        shape = [s for i, s in enumerate(self.shape) if i not in axis or self.shape[i] != 1]
        if shape == self.shape:
            return self
        return self.reshape(shape)

    def storage(self):
        data = np.empty((self.buffer.size // self.dtype().itemsize,), dtype=self.dtype)
        cl.enqueue_copy(cl_queue, data, self.buffer, is_blocking=True)
        return data

    def fill(self, value):
        cl.enqueue_fill_buffer(cl_queue, self.buffer, self.dtype(value), 0, self.size).wait()
        return self


    def transpose(self, axes):
        inst = copy.copy(self)
        inst.strides = tuple(inst.strides[a] for a in axes)
        inst.shape = tuple(inst.shape[a] for a in axes)
        inst.__update_contiguousness()
        return inst

    def __update_contiguousness(self):
        strides = [self.strides[i] for i in range(self.ndim) if self.shape[i] != 1]
        sorted_strides = sorted(strides)
        self.__f_contiguous = sorted_strides == strides
        self.__c_contiguous = sorted_strides[::-1] == strides

    @property
    def T(self):
        axes = tuple(range(len(self.shape))[::-1])
        return self.transpose(axes=axes)

    def sum(self, axis=None, keepdims=False):
        if axis is not None: assert self.__c_contiguous, "reduce_sum along axis requires c_contiguous!"
        return reduce_op("sum", self, axis=axis, keepdims=keepdims)

    def max(self, axis=None, keepdims=False):
        if axis is not None: assert self.__c_contiguous, "reduce_max along axis requires c_contiguous!"
        return reduce_op("max", self, axis=axis, keepdims=keepdims)

    def relu(self, inplace=False):
        return unary_op("relu", self, ret=self if inplace else None)

    def exp(self):
        return unary_op("exp", self)

    def log(self):
        return unary_op("log", self)

    def gt(self, value):
        return unary_op("sign", self, val=value)


class CPUArray:

    def relu(self, inplace=False):
        pass

    def gt(self, value):
        pass

