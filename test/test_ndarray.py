import runtime_path  # isort:skip

import numpy as np
from core.ndarray import GPUArray
from core.ops_gpu import unary_op

np.random.seed(0)

rnd = lambda shape: np.random.normal(0, 1, shape).astype(np.float32)

def check_array(myarr, nparr, atol=0, rtol=1e-3, ignore=()):
    assert myarr.shape == nparr.shape
    assert myarr.dtype == nparr.dtype
    if "stride" not in ignore:
        np_strides = tuple(s // myarr.dtype().itemsize for s in nparr.strides)
        assert myarr.strides == np_strides
    if "contig" not in ignore:
        assert myarr.c_contiguous == nparr.flags.c_contiguous
        assert myarr.f_contiguous == nparr.flags.f_contiguous
    assert np.allclose(myarr.numpy(), nparr, atol=atol, rtol=rtol)

def test_resahpe():
    shape = (2, 3, 4)
    nparr = np.arange(np.prod(shape)).reshape(shape).astype(np.float32)
    arr = GPUArray(nparr)
    check_array(arr, nparr)
    for shape in ((4, 3, 2), (1, 2, 3, 4), (1, 24), (24,), (3, -1)):
        check_array(arr.reshape(shape), nparr.reshape(shape))

    for shape in ((4, 3, 2), (1, 2, 3, 4), (1, 24), (24,), (3, -1)):
        check_array(arr.T.reshape(shape), nparr.T.reshape(shape, order="A"))

    for shape in ((4, 3, 2), (1, 2, 3, 4), (1, 24), (24,), (3, -1)):
        check_array(arr.transpose((0, 2, 1)).reshape(shape),
                    nparr.transpose((0, 2, 1)).reshape(shape, order="A"))

def test_contiguous():
    shape = (2, 3, 4)
    nparr = np.arange(np.prod(shape)).reshape(shape).astype(np.float32)
    arr = GPUArray(nparr)
    check_array(arr, nparr)

    arr = arr.transpose((0, 2, 1))
    nparr = nparr.transpose((0, 2, 1))
    check_array(arr, nparr)

    arr = arr.contiguous()
    nparr = np.ascontiguousarray(nparr)
    check_array(arr, nparr)

def test_expand():
    shape = (3, 1, 1)
    nparr = np.arange(np.prod(shape)).reshape(shape).astype(np.float32)
    arr = GPUArray(nparr)

    arr_expand = arr.expand((3, 3, 1))
    nparr_expand = np.tile(nparr, (1, 3, 1))
    assert np.allclose(arr_expand.numpy(), nparr_expand)

    arr_expand = arr.expand((3, 3, 3))
    nparr_expand = np.tile(nparr, (1, 3, 3))
    assert np.allclose(arr_expand.numpy(), nparr_expand)

def test_transpose():
    shape = (2, 3, 4)
    nparr = np.arange(np.prod(shape)).reshape(shape).astype(np.float32)
    arr = GPUArray(nparr)
    check_array(arr.T, nparr.T)
    check_array(arr.transpose((0, 2, 1)), nparr.transpose((0, 2, 1)))

def test_storage():
    shape = (2, 1, 3)
    nparr = np.arange(np.prod(shape)).reshape(shape).astype(np.float32)
    arr = GPUArray(nparr)
    storage = np.arange(np.prod(shape))
    assert np.allclose(arr.storage(), storage)
    # expand/tranpose should not change storage of array
    arr2 = arr.expand((2, 3, 3))
    assert np.allclose(arr2.storage(), storage)
    arr2 = arr.transpose((0, 2, 1))
    assert np.allclose(arr2.storage(), storage)

def test_squeeze():
    shape = (1, 2, 3, 1)
    nparr = rnd(shape)
    arr = GPUArray(nparr)
    check_array(arr.squeeze(), nparr.squeeze())
    check_array(arr.squeeze(axis=0), nparr.squeeze(axis=0))
    check_array(arr.squeeze(axis=-1), nparr.squeeze(axis=-1))
    check_array(arr.squeeze(axis=(0, -1)), nparr.squeeze(axis=(0, -1)))
    shape = (1, 1)
    nparr = rnd(shape)
    arr = GPUArray(nparr)
    check_array(arr.squeeze(), nparr.squeeze())

def test_unary_op():
    shape = (2, 4, 5)
    nparr = rnd(shape)
    arr = GPUArray(nparr)
    check_array(unary_op("sign", arr), np.sign(nparr).astype(np.float32))
    check_array(unary_op("neg", arr), -nparr)
    check_array(unary_op("log", arr+1e8), np.log(nparr+1e8))
    check_array(unary_op("exp", arr), np.exp(nparr))
    check_array(unary_op("relu", arr), nparr*(nparr>0))

    shape = (1,)
    nparr = rnd(shape)
    arr = GPUArray(nparr)
    arr = arr.reshape((1, 1, 1)).expand((3, 4, 5))
    nparr = np.broadcast_to(nparr.reshape((1, 1, 1)), (3, 4, 5))
    arr2 = unary_op("exp", arr)
    assert arr2.shape == arr.shape
    assert arr2.strides == arr.strides
    assert arr2.size == arr.size
    check_array(unary_op("exp", arr), np.exp(nparr), ignore=("stride", "contig"))

def test_reduce_op():
    for name in ("sum", "max"):
        for shape in [
                (1,),
                (2**6+1,),
                (2**6, 2**6+1),
                (2**6, 2**6+1, 2, 2),
                (1, 1, 1, 1),
            ]:
            nparr = np.arange(np.prod(shape)).reshape(shape).astype(np.float32)
            arr = GPUArray(nparr)
            op1, op2 = getattr(arr, name), getattr(nparr, name)
            check_array(op1(), op2())
            for axis in range(nparr.ndim):
                check_array(op1(axis=axis), op2(axis=axis))
                check_array(op1(axis=axis, keepdims=True), op2(axis=axis, keepdims=True), ignore=("stride",))

def test_fill_op():
    shape = (12, 13)
    nparr = np.empty(shape, dtype=np.float32)
    arr = GPUArray.empty(shape, dtype=np.float32)
    for val in (0, 1, 0.1):
        nparr.fill(val)
        arr.fill(val)
        check_array(arr, nparr)

    nparr = np.full(shape, 1, dtype=np.float32)
    arr = GPUArray.full(shape, 1, dtype=np.float32)
    check_array(arr, nparr)

def test_random():
    arr = GPUArray.uniform(-1, 1, (100000,))
    data = arr.numpy()
    assert np.abs(data.mean() - 0) < 1e-2

def test_broadcast():
    for shape1, shape2 in (
            [(), (1, 2, 3, 4)],
            [(1,), (1, 2, 3, 4)],
            [(1, 1, 1, 1), (1, 2, 3, 4)],
            [(4,), (1, 2, 3, 4)],
            [(3, 1), (1, 2, 3, 4)],
            [(1, 3, 1), (1, 2, 3, 4)],
            [(1, 2, 1, 1), (1, 2, 3, 4)],
            [(1,), (1,)]):
        arr1, arr2 = GPUArray.empty(shape1), GPUArray.empty(shape2)
        assert (arr1+arr2).shape == arr2.shape

def test_comparison_operators():
    rndint = lambda s: np.random.randint(0, 10, size=s).astype(np.float32)
    shape = (64, 64)
    nparr1, nparr2 = rndint(shape), rndint(shape)
    arr1, arr2 = GPUArray(nparr1), GPUArray(nparr2)
    check_array(arr1==arr2, (nparr1==nparr2).astype(np.float32))
    check_array(arr1>arr2, (nparr1>nparr2).astype(np.float32))
    check_array(arr1>=arr2, (nparr1>=nparr2).astype(np.float32))
    check_array(arr1<arr2, (nparr1<nparr2).astype(np.float32))
    check_array(arr1<=arr2, (nparr1<=nparr2).astype(np.float32))

def test_matmul_op():
    shape_pairs = [
        [(4, 5), (5, 3)],
        [(5,), (5, 3)],
        [(4, 5), (5,)],
        [(5,), (5,)],
        [(2, 4, 5), (2, 5, 3)],
        [(2, 4, 5), (1, 5, 3)],
        [(2, 4, 5), (5, 3)],
        [(2, 4, 5), (5,)],
        [(2, 3, 4, 5), (2, 3, 5, 3)],
        [(2, 3, 4, 5), (1, 1, 5, 3)],
        [(2, 3, 4, 5), (5,)],
        [(1, 32, 32), (1, 32, 32)],
        [(1, 128, 256), (1, 256, 8)],
        [(1, 4096, 256), (1, 256, 512)]
    ]
    for s1, s2 in shape_pairs:
        nparr1, nparr2 = rnd(s1), rnd(s2)
        arr1, arr2 = GPUArray(nparr1), GPUArray(nparr2)
        check_array(arr1@arr2, nparr1@nparr2, rtol=1e-3)

    s1, s2 = (4, 5), (3, 5)
    nparr1, nparr2 = rnd(s1), rnd(s2)
    arr1, arr2 = GPUArray(nparr1), GPUArray(nparr2)
    arr2, nparr2 = arr2.T, nparr2.T
    check_array(arr1@arr2, nparr1@nparr2, rtol=1e-3)

    s1, s2 = (4, 5), (1, 3)
    nparr1, nparr2 = rnd(s1), rnd(s2)
    arr1, arr2 = GPUArray(nparr1), GPUArray(nparr2)
    arr2 = arr2.expand((5, 3))
    nparr2 = np.ascontiguousarray(np.broadcast_to(nparr2, (5, 3)))
    check_array(arr1@arr2, nparr1@nparr2, rtol=1e-3)

