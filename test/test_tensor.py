import runtime_path  # isort:skip

import numpy as np
from core.tensor import Tensor

np.random.seed(0)

allclose = lambda a,b: np.allclose(a,b,equal_nan=True)

def test_binary():
    shape = (1, 2, 3)
    rnd = lambda shape: np.abs(np.random.normal(1, 1, shape)).astype(np.float32)
    ops = ("add", "sub", "mul", "truediv", "pow")
    ops = ("add", )
    backends = ("cpu", "gpu")
    for backend in backends:
        for op in ops:
            print(op, backend)
            ls, rs, inplace = f"__{op}__", f"__r{op}__", f"__i{op}__"
            npa, npb = rnd(shape), rnd(shape)
            a, b = Tensor(npa), Tensor(npb)
            a, b = getattr(a, backend)(), getattr(b, backend)()
            assert allclose(getattr(a, ls)(b).numpy(), getattr(npa, ls)(npb))
            assert allclose(getattr(a, rs)(b).numpy(), getattr(npa, rs)(npb))
            assert allclose(getattr(a, inplace)(b).numpy(), getattr(npa, inplace)(npb))


def test_unary():
    pass
