from functools import lru_cache

import os
import numpy as np
import pyopencl as cl
from pyopencl.clrandom import PhiloxGenerator as RNG
from utils.math import prod

import warnings
warnings.filterwarnings("ignore")
DEBUG = int(os.getenv("DEBUG", "0"))
OPT = int(os.getenv("OPT", "0"))

cl_ctx, cl_queue = None, None
devices = cl.get_platforms()[0].get_devices(device_type=cl.device_type.GPU)
if len(devices) == 0:
    devices = cl.get_platforms()[0].get_devices(device_type=cl.device_type.CPU)
cl_ctx = cl.Context(devices=devices)
cl_queue = cl.CommandQueue(cl_ctx)  # TODO: create one queue for every device
cl_rng = RNG(cl_ctx)

@lru_cache(maxsize=None)
def cl_build(name, program, options=tuple()):
    if DEBUG: print(f"[DEBUG] miss cache. build {name}")
    if DEBUG: print(f"[DEBUG] program {name}: \n {program}")
    cl_kernel = cl.Program(cl_ctx, program).build(tuple(options)).__getattr__(name)
    return lambda *args: cl_kernel(cl_queue, *args)

def alloc_buffer(shape, dtype, hostbuf=None):
    size = int(dtype().itemsize * prod(shape))
    flags = cl.mem_flags.READ_WRITE
    if hostbuf is not None:
        flags |= cl.mem_flags.COPY_HOST_PTR
    return cl.Buffer(cl_ctx, flags, size, hostbuf=hostbuf)

def broadcast(a, b):
    # https://numpy.org/doc/stable/user/basics.broadcasting.html
    if a.shape == b.shape:
        return a, b
    for i, j in zip(a.shape[::-1], b.shape[::-1]):
        if i != j and (i != 1) and (j != 1):
            raise ValueError(f"Error broadcasting for {a.shape} and {b.shape}")
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

def unary_op(name, a, ret=None, **kwargs):
    if ret is None:
        ret = a.__class__.copy_with_new_buffer(a) if OPT else a.__class__(shape=a.shape, dtype=a.dtype)

    code_map = {"neg": "-a", "log": "log(a)", "exp": "exp(a)", "relu": "max(a, 0.0f)", "sign": "sign(a)"}
    unary_op = cl_build("unary_op", f"""__kernel void unary_op(
        {''.join([f'int a_s{i}, int res_s{i}, ' for i in range(a.ndim)])}
        __global const float *A, __global float *B) {{
      int a_i=0, idx=0, gl_id=get_global_id(0); int ptr=gl_id;
      {''.join([f'idx=ptr/res_s{i}; ptr%=res_s{i}; a_i+=idx*a_s{i};' for i in range(a.ndim)])}
      float a=A[a_i];
      B[gl_id]={code_map[name]};
    }}""")
    args = [np.int32(s) for ss in zip(a.strides, ret.strides) for s in ss]
    unary_op((prod(a.shape),), None, *args, a.buffer, ret.buffer)
    return ret

def binary_op(name, a, b, ret=None):
    a, b = broadcast(a, b)
    if ret is None:
        ret = a.__class__(shape=a.shape, dtype=a.dtype)
    code_map = {"add": "a+b", "sub": "a-b", "truediv": "a/b", "mul": "a*b", "pow": "pow(a,b)", "eq": "(float)isequal(a,b)", "gt": "(float)isgreater(a,b)", "ge": "(float)isgreaterequal(a,b)", "drelu": "b>0?a:0.0f"}
    binary_op = cl_build("binary_op", f"""__kernel void binary_op(
        {''.join([f'int a_s{i},int b_s{i},int res_s{i},' for i in range(a.ndim)])}
        __global const float *A, __global const float *B, __global float *C) {{
      int a_i=0, b_i=0, idx=0, gl_id=get_global_id(0); int ptr=gl_id;
      {''.join(f'idx=ptr/res_s{i}; ptr%=res_s{i}; a_i+=idx*a_s{i}; b_i+=idx*b_s{i};' for i in range(a.ndim))}
      float a=A[a_i], b=B[b_i];
      C[gl_id] = {code_map[name]};
    }}""")
    args = [np.int32(s) for ss in zip(a.strides, b.strides, ret.strides) for s in ss]
    binary_op((prod(a.shape),), None, *args, a.buffer, b.buffer, ret.buffer)
    return ret

def matmul_op(a, b):
    # https://numpy.org/doc/stable/reference/generated/numpy.matmul.html
    squeezes = []
    if a.ndim == 1: a = a.reshape((1, *a.shape)); squeezes.append(0)
    if b.ndim == 1: b = b.reshape((*b.shape, 1)); squeezes.append(-1)
    ret_shape = tuple((*a.shape[:-1], b.shape[-1]))

    if a.ndim > 3: a = a.reshape((prod(a.shape[:-2]), *a.shape[2:]))
    if b.ndim > 3: b = b.reshape((prod(b.shape[:-2]), *b.shape[2:]))
    if a.ndim == 2: a = a.reshape((1, *a.shape))
    if b.ndim == 2: b = b.reshape((1, *b.shape))
    if a.shape[0] != b.shape[0]:  # broadcasting
        assert a.shape[0] == 1 or b.shape[0] == 1
        if a.shape[0] == 1 and b.shape[0] != 1: a = a.expand((b.shape[0], *a.shape[1:]))
        if b.shape[0] == 1 and a.shape[0] != 1: b = b.expand((a.shape[0], *b.shape[1:]))
    assert a.shape[0] == b.shape[0] and a.shape[2] == b.shape[1], \
            f"invalid shape for matmul {a.shape} @ {b.shape}"

    ret = a.__class__(shape=ret_shape, dtype=a.dtype)
    src = f"""__kernel void matmul_op(int BS, int M, int N, int K,
        {''.join(f'int A_s{i}, int B_s{i},' for i in range(3))}
        __global const float *A, __global const float *B, __global float *C) {{
      int bs = get_global_id(0), m = get_global_id(1), n = get_global_id(2);
      float acc = 0.0f; int A_idx, B_idx;
      for (int k=0; k<K; k++) {{
        A_idx = bs*A_s0+m*A_s1+k*A_s2; B_idx = bs*B_s0+k*B_s1+n*B_s2;
        acc += A[A_idx] * B[B_idx];
      }}
      C[bs*M*N+m*N+n] = acc;
    }}"""
    op = cl_build("matmul_op", src)
    BS, M, K, N = prod(a.shape[:-2]), a.shape[-2], a.shape[-1], b.shape[-1]
    strides = [s for ss in zip(a.strides, b.strides) for s in ss]
    args = [np.int32(a) for a in [BS, M, N, K] + strides]
    op((BS, M, N), None, *args, a.buffer, b.buffer, ret.buffer)
    for axis in squeezes:
        ret = ret.squeeze(axis)
    return ret

def contiguous_op(x):
    if not x.ndim: return x
    ret = x.__class__(shape=x.shape, dtype=x.dtype)
    args = ",".join([f"int a{i},int b{i}" for i in range(x.ndim)])
    def_strides = ";".join([f"int _s{i}="+"*".join(f"a{j}" for j in range(i+1, x.ndim))
                               for i in range(x.ndim-1)])
    def_strides += f";int _s{x.ndim-1}=1;"
    def_indices = "".join(f"int _i{i}=curr/_s{i}; curr%=_s{i}; " for i in range(x.ndim))
    addr = "+".join([f"b{i}*_i{i}" for i in range(x.ndim)])
    src = f"""__kernel void contiguous_op({args},__global const float *A, __global float *B) {{
      int curr = get_global_id(0);
      {def_strides} {def_indices}
      B[get_global_id(0)] = A[{addr}];
    }}"""
    op = cl_build("contiguous_op", src)
    args = [np.int32(s) for ss in zip(x.shape, x.strides) for s in ss]
    op((prod(x.shape),), None, *args, x.buffer, ret.buffer)
    return ret

def reduce_op(name, x, axis=None, keepdims=True):
    code_map = {"sum": "a+b", "max": "max(a,b)", "min": "min(a,b)"}
    padval_map = {"sum": "0.0f", "max": "-INFINITY", "min": "INFINITY"}
    x_shp = x.shape
    if axis is None:
        axis, x_shp = 0, (prod(x.shape),)
    size = x_shp[axis]

    def cal_ret_shape(x_shp, axis, keepdims, grp_size, n_grps):
        if n_grps <= 1:
            ret_shape = [d for i, d in enumerate(x_shp) if i != axis]
            if keepdims: ret_shape.insert(axis, 1)
            return tuple(ret_shape)
        return tuple(n_grps if i == axis else d for i, d in enumerate(x_shp))

    grp_size = 2
    max_work_group_size = cl_queue.device.max_work_group_size
    while grp_size != max_work_group_size and grp_size < size:
        grp_size *= 2
    n_grps = (size + grp_size - 1) // grp_size
    ret_shape = cal_ret_shape(x_shp, axis, keepdims, grp_size, n_grps)
    ret = x.__class__(shape=ret_shape, dtype=x.dtype)

    # merge non-target axes
    p1 = [prod(x_shp[:axis])] if axis!=0 else []
    p2 = [prod(x_shp[axis+1:])] if axis!=len(x_shp)-1 else []
    global_size = (*p1, grp_size*n_grps, *p2)
    axis, ndim = len(p1), len(global_size)

    a = [f"gl_id_{i}" for i in range(ndim)]
    b = [f"gl_s_{i}" for i in range(ndim)]
    c = ["*".join(b[i+1:]) for i in range(ndim-1)] + ["1"]
    gl2lcl = "+".join([f"{a_}*{c_}" for a_, c_ in zip(a, c)])
    a = [(f"grp_id_{i}" if i == axis else f"gl_id_{i}") for i in range(ndim)]
    b = [f"(gl_s_{i}/grp_s_{i})" for i in range(ndim)]
    c = ["*".join(b[i+1:]) for i in range(ndim-1)] + ["1"]
    lcl2gl = "+".join([f"{a_}*{c_}" for a_, c_ in zip(a, c)])
    # NOTE: calculate offset to get the proper global index
    offset = f"gl_id_0*{'0' if axis==0 else '1' if axis==ndim-1 else 'gl_s_2'}*(gl_s_{axis}-size)"
    op = cl_build("reduce_op", f"""__kernel void reduce_op(int size, __global const float *A, __local float *B, __global float *C) {{
      {''.join([f'int gl_id_{i}=get_global_id({i});int gl_s_{i}=get_global_size({i});int grp_id_{i}=get_group_id({i});int grp_s_{i}=get_local_size({i});' for i in range(ndim)])}
      int lcl_id=get_local_id({axis});
      B[lcl_id] = gl_id_{axis}<size?A[{gl2lcl}-{offset}]:{padval_map[name]};
      barrier(CLK_LOCAL_MEM_FENCE);
      for (int stride=grp_s_{axis}>>1; stride>0; stride>>=1) {{
        float a = B[lcl_id], b = B[lcl_id+stride];
        if (lcl_id<stride) B[lcl_id]={code_map[name]};
        barrier(CLK_LOCAL_MEM_FENCE);
      }}
      if (lcl_id == 0) C[{lcl2gl}]=B[0];
    }}""")
    local_mem = cl.LocalMemory(x.dtype().itemsize * grp_size)
    local_size = tuple(grp_size if i == axis else 1 for i in range(ndim))
    op(global_size, local_size, np.int32(size), x.buffer, local_mem, ret.buffer)
    if DEBUG: print(f"[DEBUG] x_shp: {x_shp} ret_shape: {ret_shape} grp_size: {grp_size} n_grps: {n_grps} size: {size} global_size: {global_size} local_size: {local_size} axis={axis} ndim={ndim} offset={offset}")
    if n_grps > 1:
        ret = reduce_op(name, ret, axis=axis, keepdims=keepdims)  # recursive reduce (inefficient)
    return ret

