import numpy as np
import scipy.stats as stats
import pyopencl.array as cl_array
from pyopencl.clrandom import PhiloxGenerator as RNG

from core.tensor import Tensor
from core.tensor import CTX, QUEUE
rng = RNG(CTX)


def get_fans(shape):
    fan_in = shape[0] if len(shape) == 2 else np.prod(shape[1:])
    fan_out = shape[1] if len(shape) == 2 else shape[0]
    return fan_in, fan_out


class Initializer:

    def __call__(self, shape, device="cpu"):
        values = self.init(tuple(shape), device=device)
        return Tensor(values, requires_grad=True, dtype=np.float32, gpu=(device=="gpu"))

    def init(self, shape, device):
        raise NotImplementedError


class NormalInit(Initializer):

    def __init__(self, mean=0.0, std=1.0):
        self._mean = mean
        self._std = std

    def init(self, shape, device="cpu"):
        if device == "cpu":
            return np.random.normal(loc=self._mean, scale=self._std, size=shape)
        elif device == "gpu":
            pass
            # TODO


class TruncatedNormalInit(Initializer):

    def __init__(self, mean=0.0, std=1.0):
        self._tn = stats.truncnorm(- 2 * std, 2 * std, loc=mean, scale=std)

    def init(self, shape):
        return self._tn.rvs(size=shape)


class UniformInit(Initializer):

    def __init__(self, a=0.0, b=1.0):
        self._a = a
        self._b = b

    def init(self, shape):
        return np.random.uniform(low=self._a, high=self._b, size=shape)


class ConstantInit(Initializer):

    def __init__(self, val):
        self._val = val

    def init(self, shape, device):
        if device == "cpu":
            return np.full(shape=shape, fill_value=self._val, dtype=np.float32)
        elif device == "gpu":
            # TODO: slower than zero + value
            return cl_array.empty(QUEUE, shape, dtype=np.float32).fill(self._val)


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

    def init(self, shape, device):
        fan_in, fan_out = get_fans(shape)
        a = self._gain * np.sqrt(6.0 / (fan_in + fan_out))
        if device == "cpu":
            return np.random.uniform(low=-a, high=a, size=shape).astype(np.float32)
        elif device == "gpu":
            return rng.uniform(a=-a, b=a, shape=shape, dtype=np.float32, cq=QUEUE)
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

    def init(self, shape):
        fan_in, fan_out = get_fans(shape)
        std = self._gain * np.sqrt(2.0 / (fan_in + fan_out))
        return np.random.normal(loc=0.0, scale=std, size=shape)


class HeUniformInit(Initializer):
    """
    Implement the He initialization method described in
    “Delving deep into rectifiers: Surpassing human-level performance
    on ImageNet classification” He, K. et al. (2015)

    Weights will have values sampled from uniform distribution U(-a, a) where
    a = sqrt(6.0 / num_in)
    """

    def __init__(self, gain=1.0):
        self._gain = gain

    def init(self, shape):
        fan_in, _ = get_fans(shape)
        a = self._gain * np.sqrt(6.0 / fan_in)
        return np.random.uniform(low=-a, high=a, size=shape)


class HeNormalInit(Initializer):
    """
    Implement the He initialization method described in
    “Delving deep into rectifiers: Surpassing human-level performance
    on ImageNet classification” He, K. et al. (2015)

    Weights will have values sampled from normal distribution N(0, std) where
    std = sqrt(2.0 / num_in)
    """

    def __init__(self, gain=1.0):
        self._gain = gain

    def init(self, shape):
        fan_in, _ = get_fans(shape)
        std = self._gain * np.sqrt(2.0 / fan_in)
        return np.random.normal(loc=0.0, scale=std, size=shape)
