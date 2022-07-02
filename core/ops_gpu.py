from functools import lru_cache

import os

import numpy as np
import pyopencl as cl

from utils.math import prod

import warnings
warnings.filterwarnings("ignore")
DEBUG = int(os.getenv("DEBUG", "0"))

# init opencl
cl_ctx, cl_queue = None, None
devices = cl.get_platforms()[0].get_devices(device_type=cl.device_type.GPU)
if len(devices) == 0:
    devices = cl.get_platforms()[0].get_devices(device_type=cl.device_type.CPU)
cl_ctx = cl.Context(devices=devices)  # TODO: cache_dir?
cl_queue = cl.CommandQueue(cl_ctx)


@lru_cache()
def cl_build(name, program, options=tuple()):
    if DEBUG: print(f"miss cache. build {name}")
    if DEBUG: print(program)
    cl_kernel = cl.Program(cl_ctx, program).build(tuple(options)).__getattr__(name)
    return lambda *args: cl_kernel(cl_queue, *args)


def alloc_buffer(shape, dtype, hostbuf=None):
    size = int(dtype().itemsize * prod(shape))
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
    code_map = {"neg": "-a", "log": "log(a)", "exp": "exp(a)", "relu": ""}  # TODO: relu
    unary_op = cl_build("unary_op", """
    __kernel void unary_op(""" +
    "".join([f"int a_s{i}, int res_s{i}, " for i in range(a.ndim)]) +
    """__global const float *A, __global float *B) {
      int res_i = 0, a_i = 0;""" +
      "".join([f"int idx{i}=get_global_id({i}); res_i+=idx{i}*res_s{i}; a_i+=idx{i}*a_s{i};" for i in range(a.ndim)]) +
    """
      float a = A[a_i];
      B[res_i] = """ + code_map[name] + """;
    }
    """)
    args = [np.int32(s) for ss in zip(a.strides, ret.strides) for s in ss]
    unary_op(a.shape, None, *args, a.buffer, ret.buffer)
    return ret


def binary_op(name, a, b, ret=None):
    a, b = broadcast(a, b)
    if ret is None:
        ret = a.__class__(shape=a.shape, dtype=a.dtype)
    code_map = {"add": "a+b", "sub": "a-b", "truediv": "a/b", "mul": "a*b", "pow": "power(a,b)"}
    binary_op = cl_build("binary_op", """
    __kernel void binary_op(""" +
    "".join([f"int a_s{i}, int b_s{i}, int res_s{i}, " for i in range(a.ndim)]) +
    """ __global const float *A, __global const float *B, __global float *C) {
      int res_i = 0, a_i = 0, b_i = 0;""" +
      "".join([f"const int idx{i} = get_global_id({i}); res_i += idx{i}*res_s{i}; a_i += idx{i}*a_s{i}; b_i += idx{i}*b_s{i};" for i in range(a.ndim)]) +
      """
      float a = A[a_i], b = B[b_i];
      C[res_i] = """ + code_map[name] + """;
    }
    """)
    args = [np.int32(s) for ss in zip(a.strides, b.strides, ret.strides) for s in ss]
    global_size = (1,) if not a.shape else a.shape
    binary_op(global_size, None, *args, a.buffer, b.buffer, ret.buffer)
    return ret


def matmul_op(a, b):
    a_, b_ = a, b
    if a.ndim == 1: a_ = a.reshape((1, *a.shape))
    if b.ndim == 1: b_ = b.reshape((*b.shape, 1))
    ret_shape = tuple((*a_.shape[:-1], b_.shape[-1]))

    if a_.ndim > 3: a_ = a_.reshape((prod(a_.shape[:-2]), *a_.shape[2:]))
    if b_.ndim > 3: b_ = b_.reshape((prod(b_.shape[:-2]), *b_.shape[2:]))
    if a_.ndim == 2: a_ = a_.reshape((1, *a_.shape))
    if b_.ndim == 2: b_ = b_.reshape((1, *b_.shape))
    if a_.shape[0] != b_.shape[0]:  # broadcasting
        assert a_.shape[0] == 1 or b_.shape[0] == 1
        if a_.shape[0] == 1: a_ = a_.expand((b_.shape[0], *a_.shape[1:]))
        if b_.shape[0] == 1: b_ = b_.expand((a_.shape[0], *b_.shape[1:]))

    ret = a.__class__(shape=ret_shape, dtype=a.dtype)
    src = """__kernel void matmul_op(int BS, int M, int N, int K,
        """ + "".join(f"int A_s{i}, int B_s{i}," for i in range(3)) + """
        __global const float *A, __global const float *B, __global float *C) {
      int bs = get_global_id(0), m = get_global_id(1), n = get_global_id(2);
      float acc = 0.0f; int A_idx, B_idx;
      for (int k=0; k<K; k++) {
        A_idx = bs*A_s0+m*A_s1+k*A_s2; B_idx = bs*B_s0+k*B_s1+n*B_s2;
        acc += A[A_idx] * B[B_idx];
      }
      C[bs*M*N+m*N+n] = acc;
    }"""
    op = cl_build("matmul_op", src)
    BS, M, K, N = prod(a_.shape[:-2]), a_.shape[-2], a_.shape[-1], b_.shape[-1]
    strides = [s for ss in zip(a_.strides, b_.strides) for s in ss]
    args = [np.int32(a_) for a_ in [BS, M, N, K] + strides]
    op((BS, M, N), None, *args, a_.buffer, b_.buffer, ret.buffer).wait()
    if a.ndim == 1: ret = ret.squeeze(axis=0)
    if b.ndim == 1: ret = ret.squeeze(axis=-1)
    return ret


def contiguous_op(x):
    if not x.ndim: return x
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
    args = [np.int32(s) for ss in zip(x.shape, x.strides) for s in ss]
    op((prod(x.shape),), None, *args, x.buffer, ret.buffer)
    return ret


def reduce_op(name, x, axis=None, keepdims=True):
    code_map = {"sum": "a+b", "max": "max(a,b)"}
    x_shp = x.shape
    if axis is None:
        axis, x_shp = 0, (prod(x.shape),)
    size = x_shp[axis]

    def cal_ret_shape(x_shp, axis, keepdims, grp_size):
        if x_shp[axis] // grp_size <= 1:
            if keepdims:
                ret_shape = (d if i!=axis else 1 for i,d in enumerate(x_shp))
            else:
                ret_shape = (d for i,d in enumerate(x_shp) if i!=axis)
        else:
            ret_shape = (d // grp_size if i == axis else d for i, d in enumerate(x_shp))
        return tuple(ret_shape)

    grp_size = 2 ** [i for i in range(8, -1, -1) if size % (2**i) == 0][0]
    assert (size & (size-1) == 0) and size != 0, f"size({size}) is not a power of 2."
    ret_shape = cal_ret_shape(x_shp, axis, keepdims, grp_size)
    ret = x.__class__(shape=ret_shape, dtype=x.dtype)

    # merge non-target axes
    if axis is not None:
        p1 = [prod(x_shp[:axis])] if axis != 0 else []
        p2 = [prod(x_shp[axis+1:])] if axis != len(x_shp) - 1 else []
        global_size = p1 + [size] + p2
        axis, ndim = len(p1), len(global_size)
        if DEBUG: print(f"\nafter merge x_shp={global_size} axis={axis} ndim={ndim}")

    a = [(f"grp_id_{i}" if i == axis else f"gl_id_{i}") for i in range(ndim)]
    b = [f"(gl_s_{i}/grp_s_{i})" for i in range(ndim)]
    c = ["*".join(b[i+1:]) for i in range(ndim-1)] + ["1"]
    lcl2gl = "+".join([f"{a_}*{c_}" for a_, c_ in zip(a, c)])
    a = [f"gl_id_{i}" for i in range(ndim)]
    b = [f"gl_s_{i}" for i in range(ndim)]
    c = ["*".join(b[i+1:]) for i in range(ndim-1)] + ["1"]
    gl2lcl = "+".join([f"{a_}*{c_}" for a_, c_ in zip(a, c)])
    op = cl_build("reduce_op", """
    __kernel void reduce_op(__global const float *A, __local float *B, __global float *C) {
      """ + "".join([
          f"int gl_id_{i}=get_global_id({i});int gl_s_{i}=get_global_size({i});int grp_id_{i}=get_group_id({i});int grp_s_{i}=get_local_size({i});\n" for i in range(ndim)]) + """
    """ + f"int lcl_id=get_local_id({axis}); B[lcl_id] = A[{gl2lcl}];" + """
      barrier(CLK_LOCAL_MEM_FENCE);
      """ + f"for (int stride=grp_s_{axis}>>1; stride>0; stride>>=1) {{" + """
        float a = B[lcl_id], b = B[lcl_id+stride];
        if (lcl_id < stride)
          B[lcl_id] = """ + code_map[name] + """;
        barrier(CLK_LOCAL_MEM_FENCE);
      }
      if (lcl_id == 0) """ + f"C[{lcl2gl}] = B[0];" + """
    }""")
    local_mem = cl.LocalMemory((x.dtype().itemsize * size) // grp_size)
    local_size = (grp_size if i == axis else 1 for i in range(ndim))
    op(global_size, tuple(local_size), x.buffer, local_mem, ret.buffer)
    if DEBUG: print(f"grp_size: {grp_size}, n_grps: {size // grp_size} retshape: {ret.shape}")
    # inefficient recursive call
    if size // grp_size > 1:
        ret = reduce_op(name, ret, axis=axis, keepdims=keepdims)
    return ret

