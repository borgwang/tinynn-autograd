from functools import lru_cache

import numpy as np
import pyopencl as cl

# init opencl
cl_ctx, cl_queue = None, None
devices = cl.get_platforms()[0].get_devices(device_type=cl.device_type.GPU)
if len(devices) == 0:
    devices = cl.get_platforms()[0].get_devices(device_type=cl.device_type.CPU)
cl_ctx = cl.Context(devices=devices)  # TODO: cache_dir?
cl_queue = cl.CommandQueue(cl_ctx)


@lru_cache()
def cl_build(name, program):
    print(f"miss cache. build {name}")
    cl_kernel = cl.Program(cl_ctx, program).build().__getattr__(name)
    return lambda *args: cl_kernel(cl_queue, *args)


def alloc_buffer(shape, dtype, hostbuf=None):
    size = int(dtype().itemsize * np.prod(shape))
    flags = cl.mem_flags.READ_WRITE
    if hostbuf is not None:
        flags |= cl.mem_flags.COPY_HOST_PTR
    return cl.Buffer(cl_ctx, flags, size, hostbuf=hostbuf)


def broadcast(a, b):
    if a.shape == b.shape:
        return a, b
    for i, j in zip(a.shape, b.shape):
        if i != j and (i != 1) and (j != 1):
            raise ValueError("Error broadcasting for {a.shape} and {b.shape}")
    ndims = max(len(a.shape), len(b.shape))
    if len(a.shape) != ndims:
        a = a.reshape([1] * (ndims - len(a.shape)) + list(a.shape))
    if len(b.shape) != ndims:
        b = b.reshape([1] * (ndims - len(b.shape)) + list(b.shape))
    broadcast_shape = [max(i, j) for i, j in zip(a.shape, b.shape)]
    if a.shape != broadcast_shape:
        a = a.expand(broadcast_shape)
    if b.shape != broadcast_shape:
        b = b.expand(broadcast_shape)
    return a, b

def unary_op(name, a, ret=None):
    if ret is None:
        ret = a.__class__(shape=a.shape, dtype=a.dtype)
    op_mapping = {"neg": "-a", "log": "log(a)", "exp": "exp(a)", "relu": ""}  # TODO: relu?
    unary_op = cl_build("unary_op", """
    __kernel void unary_op(""" +
    "".join([f"int a_s{i}, int res_s{i}, " for i in range(len(a.strides))]) +
    """__global const float *a_g, __global float *res_g) {
      int res_i = 0, a_i = 0;""" +
      "".join([f"int idx{i}=get_global_id({i}); res_i+=idx{i}*res_s{i}; a_i+=idx{i}*a_s{i};" for i in range(len(a.strides))]) +
    """
      float a = a_g[a_i];
      res_g[res_i] = """ + op_mapping[name] + """;
    }
    """)
    args = [np.int32(s) for ss in zip(a.strides, ret.strides) for s in ss]
    unary_op(a.shape, None, *args, a.buffer, ret.buffer)
    return ret

def binary_op(name, a, b, ret=None):
    a, b = broadcast(a, b)
    if ret is None:
        ret = a.__class__(shape=a.shape, dtype=a.dtype)
    op_mapping = {"add": "a+b", "sub": "a-b", "truediv": "a/b", "mul": "a*b", "pow": "power(a,b)"}
    binary_op = cl_build("binary_op", """
    __kernel void binary_op(""" +
    "".join([f"int a_s{i}, int b_s{i}, int res_s{i}, " for i in range(len(a.strides))]) +
    """ __global const float *a_g, __global const float *b_g, __global float *res_g) {
      int res_i = 0, a_i = 0, b_i = 0;""" +
      "".join([f"const int idx{i} = get_global_id({i}); res_i += idx{i}*res_s{i}; a_i += idx{i}*a_s{i}; b_i += idx{i}*b_s{i};" for i in range(len(a.strides))]) +
      """
      float a = a_g[a_i], b = b_g[b_i];
      res_g[res_i] = """ + op_mapping[name] + """;
    }
    """)
    args = [np.int32(s) for ss in zip(a.strides, b.strides, ret.strides) for s in ss]
    binary_op(a.shape, None, *args, a.buffer, b.buffer, ret.buffer)
    return ret


def matmul_op(a, b, ret=None):
    ret_shape = list(a.shape)[:-1] + list(b.shape)[1:]
    if ret is None:
        ret = a.__class__(shape=ret_shape, dtype=a.dtype)
    src = """
    __kernel void matmul_op(const int M, const int N, const int K,
        const int A_c_conus, const int B_c_conus,
        __global const float *A, __global const float *B, __global float *C) {
      const int m = get_global_id(0);
      const int n = get_global_id(1);
      float acc = 0.0f;
      int A_idx, B_idx;
      for (int k=0; k<K; k++) {
        A_idx = A_c_conus ? m*K + k : k*M + m;
        B_idx = B_c_conus ? k*N + n : n*K + k;
        acc += A[A_idx] * B[B_idx];
      }
      C[m * N + n] = acc;
    }
    """
    op = cl_build("matmul_op", src)
    M = int(np.prod(list(a.shape)[:-1]))
    K = a.shape[-1]
    N = int(np.prod(list(b.shape)[1:]))
    op((M, N), None, *[np.int32(a) for a in [M, N, K, a._c_contiguous, b._c_contiguous]],
        a.buffer, b.buffer, ret.buffer)
    return ret

def reduce_op(name, a, ret=None, axis=None, keepdims=True):
    # TODO: support axis and keepdims
    # TODO: https://github.com/JimMadge/OpenCL-Reduction-Example/blob/master/reduction/reduction.cl
    group_size = 256
    n_groups = a.shape[0] // group_size
    ret_shape = (n_groups,) if n_groups > 1 else ()
    if ret is None:
        ret = a.__class__(shape=ret_shape, dtype=a.dtype)
    op_mapping = {"sum": "", "max": ""}
    op = cl_build("reduce_op", """
    __kernel void reduce_op(
        __global const float *a, __local float *b, __global float *res_g) {
      int global_id = get_global_id(0);
      int local_id = get_local_id(0);
      int group_size = get_local_size(0);
      int group_id = get_group_id(0);
      // copy to local
      b[local_id] = a[global_id];
      barrier(CLK_LOCAL_MEM_FENCE);
      for (int stride=group_size>>1; stride>0; stride>>=1) {
        if (local_id < stride)
          b[local_id] += b[local_id + stride];
        barrier(CLK_LOCAL_MEM_FENCE);
      }
      if (local_id == 0)
        res_g[group_id] = b[0];
    }
    """)
    local_mem_size = int(a.dtype().itemsize * np.prod(a.shape)) // group_size
    local_mem = cl.LocalMemory(local_mem_size)
    op((np.prod(a.shape),), (group_size,), a.buffer, local_mem, ret.buffer)
    if n_groups > 1:
        ret = reduce_op(name, ret)
    return ret
