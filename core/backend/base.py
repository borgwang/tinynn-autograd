class Array:
    def __init__(self):
        pass

    @classmethod
    def broadcast(cls, a, b):
        # rule: https://numpy.org/doc/stable/user/basics.broadcasting.html
        if a.shape == b.shape:
            return a, b
        for i, j in zip(a.shape[::-1], b.shape[::-1]):
            if i != j and (i != 1) and (j != 1):
                raise ValueError(f"Error broadcasting for {a.shape} and {b.shape}")
        ndim = max(a.ndim, b.ndim)
        if a.ndim != ndim:
            a = a.reshape([1] * (ndim - a.ndim) + list(a.shape))
        if b.ndim != ndim:
            b = b.reshape([1] * (ndim - b.ndim) + list(b.shape))
        broadcast_shape = [max(i, j) for i, j in zip(a.shape, b.shape)]
        if a.shape != broadcast_shape:
            a = a.expand(broadcast_shape)
        if b.shape != broadcast_shape:
            b = b.expand(broadcast_shape)
        return a, b
    # ##### Unary Ops #####
    def relu(self, inplace=False): raise NotImplementedError
    def exp(self): raise NotImplementedError
    def log(self): raise NotImplementedError
    def drelu(self): raise NotImplementedError

    # ##### Binary Ops #####
    def __add__(self, other): raise NotImplementedError
    def __radd__(self, other): raise NotImplementedError
    def __iadd__(self, other): raise NotImplementedError

    # ##### Reduce Ops #####
    def sum(self, axis=None, keepdims=None): raise NotImplementedError
    def min(self, axis=None, keepdims=None): raise NotImplementedError
    def max(self, axis=None, keepdims=None): raise NotImplementedError

    # ##### Movement Ops #####
    def reshape(self, shape): raise NotImplementedError
    def expand(self, shape): raise NotImplementedError
    def squeeze(self, axis=None): raise NotImplementedError
    def permute(self, axes): raise NotImplementedError

    # ##### Slice Ops #####
    def __getitem__(self, key): raise NotImplementedError
    def __setitem__(self, key, value): raise NotImplementedError

    @property
    def T(self):
        return self.permute(axes=tuple(range(self.ndim)[::-1]))

