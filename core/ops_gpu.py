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
    #print(program)
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
    # TODO: https://github.com/JimMadge/OpenCL-Reduction-Example/blob/master/reduction/reduction.cl
    # 1. axis and keepdims
    # 2. axis=None
    # 3. padding
    # 4. contiguous
    # 5. 4D tensor reduction
    # 6. dynamic group_size
    group_size = 2 ** 8
    length = np.prod(a.shape) if axis is None else a.shape[axis]
    n_groups = length // group_size
    if n_groups <= 1:
        ret_shape = tuple(d for i, d in enumerate(a.shape) if i != axis)
    else:
        ret_shape = tuple(group_size if i == axis else d
                          for i, d in enumerate(a.shape))
    if ret is None:
        ret = a.__class__(shape=ret_shape, dtype=a.dtype)
    op_mapping = {"sum": "x1+x2", "max": "max(x1,x2)"}
    #op = cl_build("reduce_op", """
    #__kernel void reduce_op(
    #    __global const float *a, __local float *b, __global float *res_g) {
    #  int gl_id = get_global_id(0);
    #  int lcl_id = get_local_id(0);
    #  int grp_s = get_local_size(0);
    #  int grp_id = get_group_id(0);
    #  // copy to local
    #  b[lcl_id] = a[gl_id];
    #  barrier(CLK_LOCAL_MEM_FENCE);
    #  for (int stride=grp_s>>1; stride>0; stride>>=1) {
    #    if (lcl_id < stride)
    #      b[lcl_id] += b[lcl_id + stride];
    #    barrier(CLK_LOCAL_MEM_FENCE);
    #  }
    #  if (lcl_id == 0)
    #    res_g[grp_id] = b[0];
    #}
    #""")
    #op = cl_build("reduce_op", """
    #__kernel void reduce_op(
    #    __global const float *a, __local float *b, __global float *res_g) {
    #  int gl_id_0 = get_global_id(0), gl_id_1 = get_global_id(1);
    #  int gl_s_1 = get_global_size(1);
    #  int lcl_id = get_local_id(0);
    #  int grp_s_0 = get_local_size(0);
    #  int grp_s_1 = get_local_size(0);
    #  int grp_id_0 = get_group_id(0);
    #  b[lcl_id] = a[gl_id_0*gl_s_1 + gl_id_1];  // 1d-sum: gl_id; 2d-sum0:
    #  barrier(CLK_LOCAL_MEM_FENCE);
    #  for (int stride=grp_s_0>>1; stride>0; stride>>=1) {
    #    if (lcl_id < stride)
    #      b[lcl_id] += b[lcl_id + stride];
    #    barrier(CLK_LOCAL_MEM_FENCE);
    #  }
    #  if (lcl_id == 0)
    #    res_g[grp_id_0*(gl_s_1/grp_s_1)+gl_id_1] = b[0];
    #}
    #""")

    #op = cl_build("reduce_op", """
    #__kernel void reduce_op(
    #    __global const float *a, __local float *b, __global float *res_g) {
    #  int gl_id_0 = get_global_id(0), gl_id_1 = get_global_id(1);
    #  int gl_s_1 = get_global_size(1);
    #  int lcl_id = get_local_id(1);

    #  int grp_id_1 = get_group_id(1);
    #  int grp_s_0 = get_local_size(0);
    #  int grp_s_1 = get_local_size(1);

    #  b[lcl_id] = a[gl_id_0*gl_s_1 + gl_id_1];
    #  barrier(CLK_LOCAL_MEM_FENCE);
    #  for (int stride=grp_s_1>>1; stride>0; stride>>=1) {
    #    if (lcl_id < stride)
    #      b[lcl_id] += b[lcl_id + stride];
    #    barrier(CLK_LOCAL_MEM_FENCE);
    #  }
    #  if (lcl_id == 0)
    #    res_g[gl_id_0*(gl_s_1/grp_s_1)+grp_id_1] = b[0];
    #}
    #""")
    a1 = [(f"grp_id_{i}" if i == axis else f"gl_id_{i}") for i in range(len(a.shape))]
    b1 = [f"(gl_s_{i}/grp_s_{i})" for i in range(len(a.shape))]
    c1 = ["*".join(b1[i+1:]) for i in range(len(a.shape)-1)] + ["1"]
    d1 = "+".join([f"{a}*{c}" for a, c in zip(a1, c1)])

    a2 = [f"gl_id_{i}" for i in range(len(a.shape))]
    b2 = [f"gl_s_{i}" for i in range(len(a.shape))]
    c2 = ["*".join(b2[i+1:]) for i in range(len(a.shape)-1)] + ["1"]
    d2 = "+".join([f"{a}*{c}" for a, c in zip(a2, c2)])

    op = cl_build("reduce_op", """
    __kernel void reduce_op(
        __global const float *a, __local float *b, __global float *res_g) {
      """ + "".join([
          f"int gl_id_{i}=get_global_id({i});"
          f"int gl_s_{i}=get_global_size({i});"
          f"int grp_id_{i}=get_group_id({i});"
          f"int grp_s_{i}=get_local_size({i});" for i in range(len(a.shape))]) +
    f"int lcl_id=get_local_id({axis});" +
    f"b[lcl_id] = a[{d2}];" + """
      barrier(CLK_LOCAL_MEM_FENCE);
      """ + f"for (int stride=grp_s_{axis}>>1; stride>0; stride>>=1)" +
      """
      {
        float x1 = b[lcl_id], x2 = b[lcl_id + stride];
        if (lcl_id < stride)
          b[lcl_id] = """ + op_mapping[name] + """;
          //b[lcl_id] += b[lcl_id + stride];
        barrier(CLK_LOCAL_MEM_FENCE);
      }
      if (lcl_id == 0)
    """ + f"res_g[{d1}] = b[0];" + """
    }""")
    local_mem_size = int(a.dtype().itemsize * a.shape[axis]) // group_size
    local_mem = cl.LocalMemory(local_mem_size)
    local_size = tuple(group_size if i == axis else 1 for i in range(len(a.shape)))
    op(a.shape, local_size, a.buffer, local_mem, ret.buffer)
    if n_groups > 1:
        ret = reduce_op(name, ret, axis=axis)
    return ret
