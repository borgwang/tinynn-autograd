import numpy as np

from core.tensor import Tensor
from env import BACKEND
from core.backend.numpy import NpArray as CPUArray
GPUArray = type(None)
if BACKEND == "opencl":
    from core.backend.opencl import ClArray as GPUArray
elif BACKEND == "cuda":
    from core.backend.cuda import CuArray as GPUArray

def get_fans(shape):
    fan_in = shape[0] if len(shape) == 2 else np.prod(shape[1:])
    fan_out = shape[1] if len(shape) == 2 else shape[0]
    return fan_in, fan_out

class Initializer:
    def __call__(self, shape, dtype=np.float32, device="cpu", name=""):
        values = self.init(tuple(shape), dtype=dtype, device=device)
        return Tensor(values, requires_grad=True, dtype=dtype, name=name)

    def init(self, shape, dtype, device):
        raise NotImplementedError

class NormalInit(Initializer):
    def __init__(self, mean=0.0, std=1.0):
        self._mean = mean
        self._std = std

    def init(self, shape, dtype, device):
        if device == "cpu":
            return CPUArray.normal(loc=self._mean, scale=self._std, shape=shape, dtype=dtype)
        elif device == "gpu":
            return GPUArray.normal(loc=self._mean, scale=self._std, shape=shape, dtype=dtype)
        else:
            raise ValueError(f"Invalid device type {device}")

class UniformInit(Initializer):
    def __init__(self, a=0.0, b=1.0):
        self._a = a
        self._b = b

    def init(self, shape, dtype, device):
        if device == "cpu":
            return CPUArray.uniform(a=self._a, b=self._b, shape=shape, dtype=dtype)
        elif device == "gpu":
            return GPUArray.uniform(a=self._a, b=self._b, shape=shape, dtype=dtype)
        else:
            raise ValueError(f"Invalid device type {device}")

class ConstantInit(Initializer):
    def __init__(self, val):
        self._val = val

    def init(self, shape, dtype, device):
        if device == "cpu":
            return CPUArray.full(shape, self._val, dtype)
        elif device == "gpu":
            return GPUArray.full(shape, self._val, dtype)
        else:
            raise ValueError(f"Invalid device type {device}")

class ZerosInit(ConstantInit):
    def __init__(self):
        super(ZerosInit, self).__init__(0.0)

class XavierUniformInit(Initializer):
    """
    Implement the Xavier method described in
    "Understanding the difficulty of training deep feedforward neural networks”
    Glorot, X. & Bengio, Y. (2010)

    Weights will have values sampled from uniform distribution U(-a, a) where
    a = gain * sqrt(6.0 / (num_in + num_out))
    """
    def __init__(self, gain=1.0):
        self._gain = gain

    def init(self, shape, dtype, device):
        fan_in, fan_out = get_fans(shape)
        a = self._gain * np.sqrt(6.0 / (fan_in + fan_out))
        if device == "cpu":
            return CPUArray.uniform(a=-a, b=a, shape=shape, dtype=dtype)
        elif device == "gpu":
            return GPUArray.uniform(a=-a, b=a, shape=shape, dtype=dtype)
        else:
            raise ValueError(f"Invalid device type {device}")

class XavierNormalInit(Initializer):
    """
    Implement the Xavier method described in
    "Understanding the difficulty of training deep feedforward neural networks”
    Glorot, X. & Bengio, Y. (2010)

    Weights will have values sampled from uniform distribution N(0, std) where
    std = gain * sqrt(1.0 / (num_in + num_out))
    """
    def __init__(self, gain=1.0):
        self._gain = gain

    def init(self, shape, dtype, device):
        fan_in, fan_out = get_fans(shape)
        std = self._gain * np.sqrt(2.0 / (fan_in + fan_out))
        if device == "cpu":
            return CPUArray.normal(loc=0.0, scale=std, shape=shape, dtype=dtype)
        elif device == "gpu":
            return GPUArray.normal(loc=0.0, scale=std, shape=shape, dtype=dtype)
        else:
            raise ValueError(f"Invalid device type {device}")
