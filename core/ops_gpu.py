from functools import lru_cache

import numpy as np
import pyopencl as cl

import warnings
warnings.filterwarnings("ignore")

# init opencl
cl_ctx, cl_queue = None, None
devices = cl.get_platforms()[0].get_devices(device_type=cl.device_type.GPU)
if len(devices) == 0:
    devices = cl.get_platforms()[0].get_devices(device_type=cl.device_type.CPU)
cl_ctx = cl.Context(devices=devices)  # TODO: cache_dir?
cl_queue = cl.CommandQueue(cl_ctx)


@lru_cache()
def cl_build(name, program):
    #print(f"miss cache. build {name}")
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


def unary_op(name, a, ret=None):
    if ret is None:
        ret = a.__class__(shape=a.shape, dtype=a.dtype)
    op_mapping = {"neg": "-a", "log": "log(a)", "exp": "exp(a)", "relu": ""}  # TODO: relu
    unary_op = cl_build("unary_op", """
    __kernel void unary_op(""" +
    "".join([f"int a_s{i}, int res_s{i}, " for i in range(a.ndim)]) +
    """__global const float *A, __global float *B) {
      int res_i = 0, a_i = 0;""" +
      "".join([f"int idx{i}=get_global_id({i}); res_i+=idx{i}*res_s{i}; a_i+=idx{i}*a_s{i};" for i in range(a.ndim)]) +
    """
      float a = A[a_i];
      B[res_i] = """ + op_mapping[name] + """;
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
    "".join([f"int a_s{i}, int b_s{i}, int res_s{i}, " for i in range(a.ndim)]) +
    """ __global const float *A, __global const float *B, __global float *C) {
      int res_i = 0, a_i = 0, b_i = 0;""" +
      "".join([f"const int idx{i} = get_global_id({i}); res_i += idx{i}*res_s{i}; a_i += idx{i}*a_s{i}; b_i += idx{i}*b_s{i};" for i in range(a.ndim)]) +
      """
      float a = A[a_i], b = B[b_i];
      C[res_i] = """ + op_mapping[name] + """;
    }
    """)
    args = [np.int32(s) for ss in zip(a.strides, b.strides, ret.strides) for s in ss]
    global_size = (1,) if not a.shape else a.shape
    binary_op(global_size, None, *args, a.buffer, b.buffer, ret.buffer)
    return ret


def matmul_op(a, b, ret=None):
    ret_shape = list(a.shape)[:-1] + list(b.shape)[1:]
    if ret is None:
        ret = a.__class__(shape=ret_shape, dtype=a.dtype)
    src = """
    __kernel void matmul_op(const int M, const int N, const int K,
        const int A_c_contig, const int B_c_contig,
        __global const float *A, __global const float *B, __global float *C) {
      int m = get_global_id(0), n = get_global_id(1);
      float acc = 0.0f;
      int A_idx, B_idx;
      for (int k=0; k<K; k++) {
        A_idx = A_c_contig ? m*K + k : k*M + m;
        B_idx = B_c_contig ? k*N + n : n*K + k;
        acc += A[A_idx] * B[B_idx];
      }
      C[m * N + n] = acc;
    }
    """
    op = cl_build("matmul_op", src)
    M = int(np.prod(list(a.shape)[:-1]))
    K = a.shape[-1]
    N = int(np.prod(list(b.shape)[1:]))
    op((M, N), None, *[np.int32(a) for a in [M, N, K, a.c_contiguous, b.c_contiguous]],
        a.buffer, b.buffer, ret.buffer)
    return ret


def contiguous_op(x):
    if not x.ndim:
        return x
    ret = x.__class__(shape=x.shape, dtype=x.dtype)
    args = "".join([f"int a{i},int b{i}," for i in range(x.ndim)])
    def_strides = ";".join([f"int _s{i}="+"*".join(f"a{j}" for j in range(i+1, x.ndim))
                               for i in range(x.ndim-1)])
    def_strides += f";int _s{x.ndim-1}=1;"
    def_indices = "".join(f"int _i{i}=curr/_s{i}; curr%=_s{i}; " for i in range(x.ndim))
    addr = "+".join([f"b{i}*_i{i}" for i in range(x.ndim)])
    src = """
    __kernel void contiguous_op(""" + args + """__global const float *A, __global float *B) {
      int curr = get_global_id(0);
      """ + def_strides + def_indices + """
      B[get_global_id(0)] = A[""" + addr + """];
    }
    """
    op = cl_build("contiguous_op", src)
    args = sum([[np.int32(a), np.int32(b)] for a, b in zip(x.shape, x.strides)], [])
    op((np.prod(x.shape),), None, *args, x.buffer, ret.buffer)
    return ret


def reduce_op(name, x, ret=None, axis=None, keepdims=True):
    # TODO: https://github.com/JimMadge/OpenCL-Reduction-Example/blob/master/reduction/reduction.cl
    # - padding
    # - handle uncontiguous input
    # - 4D tensor reduction
    x_shape = x.shape
    if axis is None:
        axis, x_shape = 0, (np.prod(x.shape),)
    ndim, length = len(x_shape), x_shape[axis]
    grp_size = 2 ** [i for i in range(8, -1, -1) if length % (2**i) == 0][0]
    n_groups = length // grp_size
    if n_groups <= 1:
        if keepdims:
            ret_shape = tuple(d if i != axis else 1 for i, d in enumerate(x_shape))
        else:
            ret_shape = tuple(d for i, d in enumerate(x_shape) if i != axis)
    else:
        ret_shape = tuple(d // grp_size if i == axis else d for i, d in enumerate(x_shape))
    if ret is None:
        ret = x.__class__(shape=ret_shape, dtype=x.dtype)
    op_mapping = {"sum": "a+b", "max": "max(a,b)"}

    a = [(f"grp_id_{i}" if i == axis else f"gl_id_{i}") for i in range(ndim)]
    b = [f"(gl_s_{i}/grp_s_{i})" for i in range(ndim)]
    c = ["*".join(b[i+1:]) for i in range(ndim-1)] + ["1"]
    lcl2gl = "+".join([f"{a_}*{c_}" for a_, c_ in zip(a, c)])
    a = [f"gl_id_{i}" for i in range(ndim)]
    b = [f"gl_s_{i}" for i in range(ndim)]
    c = ["*".join(b[i+1:]) for i in range(ndim-1)] + ["1"]
    gl2lcl = "+".join([f"{a_}*{c_}" for a_, c_ in zip(a, c)])
    op = cl_build("reduce_op", """
    __kernel void reduce_op(
        __global const float *A, __local float *B, __global float *C) {
      """ + "".join([
          f"int gl_id_{i}=get_global_id({i});int gl_s_{i}=get_global_size({i});"
          f"int grp_id_{i}=get_group_id({i});int grp_s_{i}=get_local_size({i});"
              for i in range(ndim)]) +
    f"int lcl_id=get_local_id({axis});" +
    f"B[lcl_id] = A[{gl2lcl}];" + """
      barrier(CLK_LOCAL_MEM_FENCE);
      """ + f"for (int stride=grp_s_{axis}>>1; stride>0; stride>>=1)" +
      """
      {
        float a = B[lcl_id], b = B[lcl_id+stride];
        if (lcl_id < stride)
          B[lcl_id] = """ + op_mapping[name] + """;
        barrier(CLK_LOCAL_MEM_FENCE);
      }
      if (lcl_id == 0) """ + f"C[{lcl2gl}] = B[0];" + """
    }""")
    local_mem_size = int(x.dtype().itemsize * x_shape[axis]) // grp_size
    local_mem = cl.LocalMemory(local_mem_size)
    local_size = tuple(grp_size if i == axis else 1 for i in range(ndim))
    op(x_shape, local_size, x.buffer, local_mem, ret.buffer)
    print(f"grp_size: {grp_size}, n_groups: {n_groups} retshape: {ret.shape}")
    print(f"!!!!!!!!!! {ret.numpy()}")
    if n_groups > 1:
        ret = reduce_op(name, ret, axis=axis, keepdims=keepdims)
    return ret

