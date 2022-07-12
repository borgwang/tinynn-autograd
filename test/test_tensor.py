import runtime_path  # isort:skip

import pytest
import numpy as np
from core.tensor import Tensor

np.random.seed(0)

rnd = lambda shape: np.abs(np.random.normal(1, 1, shape)).astype(np.float32)

def check_tensor(a, b, atol=0, rtol=1e-4):
    assert a.shape == b.shape
    assert np.allclose(a.numpy(), b, atol=atol, rtol=rtol, equal_nan=True)

def test_binary():
    shape = (10, 2, 3, 4)
    ops = ("add", "sub", "mul", "truediv", "pow")
    devices = ("cpu", "gpu")
    for device in devices:
        for op in ops:
            ls, rs, inplace = f"__{op}__", f"__r{op}__", f"__i{op}__"
            npa, npb = rnd(shape), rnd(shape)
            a, b = getattr(Tensor(npa), device)(), getattr(Tensor(npb), device)()
            check_tensor(getattr(a, ls)(b), getattr(npa, ls)(npb))
            check_tensor(getattr(a, rs)(b), getattr(npa, rs)(npb))
            check_tensor(getattr(a, inplace)(b), getattr(npa, inplace)(npb))

def test_unary():
    shape = (10, 2, 3)
    devices = ("gpu",)
    for device in devices:
        npa = rnd(shape)
        a = getattr(Tensor(npa), device)()
        check_tensor((-a), -npa)
        check_tensor(((a+1e8).log()), np.log(npa+1e8))
        check_tensor((a.exp()), np.exp(npa))
        check_tensor((a.relu()), npa*(npa>0))
        check_tensor((a>0), (npa>0).astype(np.float32))

def test_comparison_operators():
    shape = (64, 64)
    rndint = lambda s: np.random.randint(0, 10, size=s).astype(np.float32)
    npa, npb = rndint(shape), rndint(shape)
    for device in ("gpu",):
        a, b = getattr(Tensor(npa, requires_grad=True), device)(), getattr(Tensor(npb, requires_grad=True), device)()
        check_tensor(a==b, (npa==npb).astype(np.float32))
        check_tensor(a>b, (npa>npb).astype(np.float32))
        check_tensor(a>=b, (npa>=npb).astype(np.float32))
        check_tensor(a<b, (npa<npb).astype(np.float32))
        check_tensor(a<=b, (npa<=npb).astype(np.float32))
        # comparison operator is not differentiable
        with pytest.raises(Exception):
            (a > b).backward()
