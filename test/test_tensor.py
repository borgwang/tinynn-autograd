import runtime_path  # isort:skip

import numpy as np
from core.tensor import Tensor

np.random.seed(0)

rnd = lambda shape: np.abs(np.random.normal(1, 1, shape)).astype(np.float32)

def check_tensor(a, b, atol=0, rtol=1e-4):
    assert a.shape == b.shape
    assert a.dtype == b.dtype
    assert np.allclose(a, b, atol=atol, rtol=rtol, equal_nan=True)


def test_binary():
    shape = (10, 2, 3, 4)
    ops = ("add", "sub", "mul", "truediv", "pow")
    devices = ("cpu", "gpu")
    for device in devices:
        for op in ops:
            ls, rs, inplace = f"__{op}__", f"__r{op}__", f"__i{op}__"
            npa, npb = rnd(shape), rnd(shape)
            a, b = getattr(Tensor(npa), device)(), getattr(Tensor(npb), device)()
            check_tensor(getattr(a, ls)(b).numpy(), getattr(npa, ls)(npb))
            check_tensor(getattr(a, rs)(b).numpy(), getattr(npa, rs)(npb))
            check_tensor(getattr(a, inplace)(b).numpy(), getattr(npa, inplace)(npb))

def test_unary():
    shape = (10, 2, 3)
    devices = ("gpu",)
    for device in devices:
        npa = rnd(shape)
        a = getattr(Tensor(npa), device)()
        check_tensor((-a).numpy(), -npa)
        check_tensor(((a+1e8).log()).numpy(), np.log(npa+1e8))
        check_tensor((a.exp()).numpy(), np.exp(npa))
        check_tensor((a.relu()).numpy(), npa*(npa>0))
        check_tensor((a>0).numpy(), (npa>0).astype(np.float32))

