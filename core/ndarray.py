import numpy as np
import pyopencl as cl

from core.ops_gpu import cl_ctx, cl_queue, alloc_buffer
from core.ops_gpu import binary_op, matmul_op, unary_op, reduce_op


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
        self._c_contiguous = True
        self._f_contiguous = False

        # TODO: dynamic overloading
        #for op in ("mul", "add", "sub", "truediv"):
        #    setattr(self, f"__{op}__", lambda x: binary_op(op, self, as_gpu_array(x)))
        #    setattr(self, f"__i{op}__", lambda x: binary_op(op, self, as_gpu_array(x), ret=self))
        #    setattr(self, f"__r{op}__", lambda x: binary_op(op, as_gpu_array(x), self))

    @property
    def size(self):
        return int(self.dtype().itemsize * np.prod(self.shape))

    @classmethod
    def empty(cls, shape, dtype):
        return cls(shape=shape, dtype=dtype)

    @classmethod
    def zeros(cls, shape, dtype=np.float32):
        arr = cls(shape=shape, dtype=dtype)
        arr._fill(0.0)
        return arr

    @classmethod
    def ones(cls, shape, dtype=np.float32):
        arr = cls(shape=shape, dtype=dtype)
        arr._fill(1.0)
        return arr

    @classmethod
    def from_cpu(cls, buffer_data):
        return cls(data=buffer_data)

    def to_cpu(self, shape=None):
        shape = self.shape if shape is None else shape
        data = np.empty(shape, dtype=self.dtype)
        cl.enqueue_copy(cl_queue, data, self.buffer, is_blocking=True)
        return data

    def reshape(self, shape):
        # TODO: need to copy under certain conditions
        assert np.prod(shape) == np.prod(self.shape)
        # need to update strides?
        self.strides = tuple(int(np.prod(shape[i+1:])) for i in range(len(shape)))
        self.shape = shape
        return self

    def expand(self, shape):
        assert len(shape) == len(self.shape)

        strides = []
        for i, (s1, s2) in enumerate(zip(self.shape, shape)):
            if s1 < s2:
                assert s1 == 1
                stride = 0
            else:
                stride = self.strides[i]
            strides.append(stride)
        self.shape = tuple(shape)
        self.strides = tuple(strides)
        return self

    def storage(self):
        return self.to_cpu(shape=(self.buffer.size // self.dtype().itemsize,))

    def _fill(self, value):
        cl.enqueue_fill_buffer(cl_queue, self.buffer, self.dtype(value), 0, self.size)

    def __add__(self, x):
        return binary_op("add", self, as_gpu_array(x))

    def __iadd__(self, x):
        return binary_op("add", self, as_gpu_array(x), ret=self)

    def __radd__(self, x):
        return binary_op("add", as_gpu_array(x), self)

    def __sub__(self, x):
        return binary_op("sub", self, as_gpu_array(x))

    def __isub__(self, x):
        return binary_op("sub", self, as_gpu_array(x), ret=self)

    def __rsub__(self, x):
        return binary_op("sub", as_gpu_array(x), self)

    def __mul__(self, x):
        return binary_op("mul", self, as_gpu_array(x))

    def __imul__(self, x):
        return binary_op("mul", self, as_gpu_array(x), ret=self)

    def __rmul__(self, x):
        return binary_op("mul", as_gpu_array(x), self)

    def __truediv__(self, x):
        return binary_op("truediv", self, as_gpu_array(x))

    def __itruediv__(self, x):
        return binary_op("truediv", self, as_gpu_array(x), ret=self)

    def __rtruediv__(self, x):
        return binary_op("truediv", as_gpu_array(x), self)

    def __matmul__(self, x):
        return matmul_op(self, as_gpu_array(x))

    def __neg__(self):
        return unary_op("neg", self)

    def transpose(self, axes):
        dim0, dim1 = axes
        tmp = list(self.strides)
        tmp[dim0], tmp[dim1] = tmp[dim1], tmp[dim0]
        self.strides = tuple(tmp)
        tmp = list(self.shape)
        tmp[dim0], tmp[dim1] = tmp[dim1], tmp[dim0]
        self.shape = tuple(tmp)
        if self._c_contiguous:
            self._c_contiguous, self._f_contiguous = False, True
        if self._f_contiguous:
            self._c_contiguous, self._f_contiguous = True, False
        return self

    def sum(self, axis):
        return reduce_op("sum", self, axis=axis)


class CPUArray:
    pass

