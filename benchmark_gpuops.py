import pyopencl as cl
import numpy as np

from core.ndarray import GPUArray
from core.ops_gpu import matmul_op

@profile
def test():
    N = 512
    shape = (N, N)
    nparr1 = np.random.normal(0, 1, shape).astype(np.float32)
    nparr2 = np.random.normal(0, 1, shape).astype(np.float32)
    arr1, arr2 = GPUArray(nparr1), GPUArray(nparr2)
    for _ in range(100):
        ret = matmul_op(arr1, arr2)

test()
