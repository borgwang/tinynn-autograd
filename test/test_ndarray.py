import runtime_path  # isort:skip

import numpy as np
from core.ndarray import GPUArray


def check_array(myarr, nparr):
    assert myarr.shape == nparr.shape  # shape
    assert myarr.dtype == nparr.dtype  # dtype
    # strides
    np_strides = tuple(s // myarr.dtype().itemsize for s in nparr.strides)
    assert myarr.strides == np_strides
    # contiguousness
    assert myarr.c_contiguous == nparr.flags.c_contiguous
    assert myarr.f_contiguous == nparr.flags.f_contiguous
    # values
    assert np.allclose(myarr.numpy(), nparr)

def test_resahpe():
    shape = (2, 3, 4)
    nparr = np.arange(np.prod(shape)).reshape(shape).astype(np.float32)
    arr = GPUArray(nparr)
    check_array(arr, nparr)
    for shape in ((4, 3, 2), (1, 2, 3, 4), (1, 24), (24,)):
        check_array(arr.reshape(shape), nparr.reshape(shape))

    # reshape uncontiguous array
    for shape in ((4, 3, 2), (1, 2, 3, 4), (1, 24), (24,)):
        check_array(arr.T.reshape(shape), nparr.T.reshape(shape, order="A"))

    for shape in ((4, 3, 2), (1, 2, 3, 4), (1, 24), (24,)):
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

def test_reduce_op():
    shape = (4, 2**8)
    nparr = np.arange(np.prod(shape)).reshape(shape).astype(np.float32)
    arr = GPUArray(nparr)
    # test for c contiguous array
    check_array(arr.sum(), nparr.sum())
    check_array(arr.sum(axis=1), nparr.sum(axis=1))
    # test for 4d array
    shape = (4, 2**8, 4, 2**4)
    nparr = np.arange(np.prod(shape)).reshape(shape).astype(np.float32)
    arr = GPUArray(nparr)
    check_array(arr.sum(), nparr.sum())
    check_array(arr.sum(axis=0), nparr.sum(axis=0))
    check_array(arr.sum(axis=3), nparr.sum(axis=3))
    check_array(arr.sum(axis=1), nparr.sum(axis=1))
    # TODO: test for nonconitguous array?

def test_matmul_op():
    rnd = lambda shape: np.random.normal(0, 1, shape).astype(np.float32)
    nparr1, nparr2 = rnd((3, 4)), rnd((4, 3))
    arr1, arr2 = GPUArray(nparr1), GPUArray(nparr2)
    check_array(arr1@arr2, nparr1@nparr2)
    check_array(arr1.T@arr2.T, nparr1.T@nparr2.T)


