from utils.dtype import float32

class Array:
    def __init__(self, shape=None, dtype=float32):
        self.shape, self.dtype = shape, dtype
        self.register_ops()

    @property
    def size(self):
        raise NotImplementedError

    @property
    def ndim(self):
        raise NotImplementedError

    @classmethod
    def asarray(cls, obj):
        if not isinstance(obj, cls):
            obj = cls(obj.numpy()) if issubclass(obj.__class__, Array) else cls(obj)
        return obj

    @classmethod
    def register_ops(cls):
        for op in ("add", "sub", "mul", "div", "pow"):
            opname = "truediv" if op is "div" else op
            setattr(cls, f"__{opname}__", \
                (lambda op: lambda a, b: getattr(a, op)(cls.asarray(b)))(op))
            setattr(cls, f"__i{opname}__", \
                (lambda op: lambda a, b: getattr(a, op)(cls.asarray(b), out=a))(op))
            setattr(cls, f"__r{opname}__", \
                (lambda op: lambda a, b: getattr(cls.asarray(b), op)(a))(op))
        for op in ("eq", "ge", "gt", "le", "lt"):
            setattr(cls, f"__{op}__", \
                (lambda op: lambda a, b: getattr(a, op)(cls.asarray(b)))(op))
        setattr(cls, f"__matmul__", lambda a, b: a.matmul(cls.asarray(b)))
        setattr(cls, f"__neg__", lambda a: a.neg())

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
    def neg(self): raise NotImplementedError
    def exp(self): raise NotImplementedError
    def log(self): raise NotImplementedError
    def relu(self): raise NotImplementedError

    # ##### Binary Ops #####
    def add(self, other, out=None): raise NotImplementedError
    def sub(self, other, out=None): raise NotImplementedError
    def mul(self, other, out=None): raise NotImplementedError
    def div(self, other, out=None): raise NotImplementedError
    def pow(self, other, out=None): raise NotImplementedError
    def eq(self, other, out=None): raise NotImplementedError
    def gt(self, other): raise NotImplementedError
    def ge(self, other): raise NotImplementedError
    def lt(self, other): raise NotImplementedError
    def le(self, other): raise NotImplementedError
    def matmul(self, other): raise NotImplementedError
    def drelu(self, other): raise NotImplementedError

    # ##### Reduce Ops #####
    def sum(self, axis=None, keepdims=False): raise NotImplementedError
    def max(self, axis=None, keepdims=False): raise NotImplementedError

    # ##### Movement Ops #####
    def reshape(self, shape): raise NotImplementedError
    def expand(self, shape): raise NotImplementedError
    def squeeze(self, axis=None): raise NotImplementedError
    def permute(self, axes): raise NotImplementedError

    @property
    def T(self):
        return self.permute(axes=tuple(range(self.ndim)[::-1]))

    # ##### Slice Ops #####
    def __getitem__(self, key): raise NotImplementedError
    def __setitem__(self, key, value): raise NotImplementedError

    # #### Contruct Ops #####
    @classmethod
    def empty(cls, shape, dtype=float32): raise NotImplementedError
    @classmethod
    def zeros(cls, shape, dtype=float32): raise NotImplementedError
    @classmethod
    def ones(cls, shape, dtype=float32): raise NotImplementedError
    @classmethod
    def full(cls, shape, dtype=float32): raise NotImplementedError
    @classmethod
    def uniform(cls, a, b, shape, dtype=float32): raise NotImplementedError
    @classmethod
    def normal(cls, loc, scale, shape, dtype=float32): raise NotImplementedError

