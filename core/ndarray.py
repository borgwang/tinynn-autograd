import copy

import numpy as np
import pyopencl as cl

from core.ops_gpu import cl_ctx, cl_queue, alloc_buffer
from core.ops_gpu import binary_op, matmul_op, unary_op, reduce_op, contiguous_op


def as_gpu_array(obj):
    if not isinstance(obj, GPUArray):
        obj = GPUArray(obj)
    return obj


class GPUArray:

    def __init__(self, data=None, shape=None, dtype=np.float32):
        if data is not None:
            data = np.asarray(data, dtype=dtype)
        shape = tuple(shape) if shape is not None else tuple(data.shape)
        self.buffer = alloc_buffer(shape, dtype, data)
        self.strides = tuple(int(np.prod(shape[i+1:])) for i in range(len(shape)))
        self.shape, self.dtype = shape, dtype
        self.__c_contiguous, self.__f_contiguous = True, False
        self.register_ops()

    def register_ops(self):
        cls = self.__class__
        for op in ("add", "sub", "mul", "truediv"):
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
    def empty(cls, shape, dtype):
        return cls(shape=shape, dtype=dtype)

    @classmethod
    def zeros(cls, shape, dtype=np.float32):
        inst = cls(shape=shape, dtype=dtype)
        inst._fill(0.0)
        return inst

    @classmethod
    def ones(cls, shape, dtype=np.float32):
        inst = cls(shape=shape, dtype=dtype)
        inst._fill(1.0)
        return inst

    @classmethod
    def from_numpy(cls, arr):
        return cls(data=arr)

    def numpy(self):
        data = np.empty(self.shape, dtype=self.dtype)
        cl.enqueue_copy(cl_queue, data, self.contiguous().buffer, is_blocking=True)
        return data

    def contiguous(self):
        return contiguous_op(self)

    def reshape(self, shape):
        assert np.prod(shape) == np.prod(self.shape)
        inst = copy.copy(self)
        if self.__c_contiguous or self.__f_contiguous:
            if self.__c_contiguous:
                inst.strides = tuple(int(np.prod(shape[i+1:])) for i in range(len(shape)))
            else:
                inst.strides = tuple(int(np.prod(shape[:i])) for i in range(len(shape)))
        else:
            # return a copy
            inst = self.contiguous()
            inst.strides = tuple(int(np.prod(shape[i+1:])) for i in range(len(shape)))
        inst.shape = tuple(shape)
        inst.__update_contiguousness()
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

    def storage(self):
        data = np.empty((self.buffer.size // self.dtype().itemsize,), dtype=self.dtype)
        cl.enqueue_copy(cl_queue, data, self.buffer, is_blocking=True)
        return data

    def _fill(self, value):
        cl.enqueue_fill_buffer(cl_queue, self.buffer, self.dtype(value), 0, self.size)

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

    def sum(self, axis, keepdims):
        return reduce_op("sum", self, axis=axis, keepdims=keepdims)

    def max(self, axis, keepdims):
        return reduce_op("max", self, axis=axis, keepdims=keepdims)


class CPUArray:
    pass

