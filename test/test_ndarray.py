import runtime_path  # isort:skip

import numpy as np
from core.ndarray import GPUArray


def check_array(myarray, nparray):
    assert myarray.shape == nparray.shape  # shape
    assert myarray.dtype == nparray.dtype  # dtype
    # strides
    strides = tuple(s * myarray.dtype().itemsize for s in myarray.strides)
    assert strides == nparray.strides
    # contiguousness
    assert myarray._c_contiguous == nparray.flags.c_contiguous
    assert myarray._f_contiguous == nparray.flags.f_contiguous
    # values
    assert np.allclose(myarray.to_cpu(), nparray)


def _test_resahpe():
    shape = (2, 3, 4)
    nparray = np.arange(np.prod(shape)).reshape(shape).astype(np.float32)
    array = GPUArray(nparray)
    check_array(array, nparray)
    for shape in ((1, 2, 3, 4), (1, 24), (24,)):
        check_array(array.reshpae(shape), nparray.reshape(shape))


def test_contiguous():
    shape = (3, 4, 3)
    nparray = np.arange(np.prod(shape)).reshape(shape).astype(np.float32)
    array = GPUArray(nparray)
    check_array(array, nparray)

    array = array.transpose((0, 2, 1))
    nparray = nparray.transpose((0, 2, 1))
    check_array(array, nparray)

    array = array.contiguous()
    nparray = np.ascontiguousarray(nparray)
    check_array(array, nparray)


def _test_expand():
    shape = (3, 1, 4)
    nparray = np.arange(np.prod(shape)).reshape(shape).astype(np.float32)
    array = GPUArray(nparray)
    array = array.expand((3, 3, 4))
    nparray = np.tile(nparray, (1, 3, 1))
    assert not (array._c_contiguous and array._f_contiguous)
    assert np.allclose(array.to_cpu(), nparray)
    import pdb; pdb.set_trace()


def test_transpose():
    shape = (2, 3, 4)
    nparray = np.arange(np.prod(shape)).reshape(shape).astype(np.float32)
    array = GPUArray(nparray)

    check_array(array.T, nparray.T)
    check_array(array.transpose((0, 2, 1)), nparray.transpose((0, 2, 1)))

