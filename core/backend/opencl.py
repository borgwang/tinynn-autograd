from functools import lru_cache
import copy
from collections import defaultdict
import numpy as np
import pyopencl

from env import DEBUG, GRAPH
from utils.math import prod
from utils.dtype import int32, float32
from core.backend.base import Array

class CLContext:
    def __init__(self):
        self.ctx, self.queue = None, None
        platform = pyopencl.get_platforms()[0]
        devices = platform.get_devices(device_type=pyopencl.device_type.GPU)
        if len(devices) == 0:
            devices = platform.get_devices(device_type=pyopencl.device_type.CPU)
        self.ctx = pyopencl.Context(devices)
        self.queue = pyopencl.CommandQueue(self.ctx)
        # random number generator
        import pyopencl.clrandom as clrandom
        self.rng = clrandom.PhiloxGenerator(self.ctx)

    @lru_cache(maxsize=None)
    def build(self, name, program):
        if DEBUG>1: print(f"[DEBUG] program {name}: \n {program}")
        kernel = pyopencl.Program(self.ctx, program).build().__getattr__(name)
        return lambda *args: kernel(self.queue, *args)

    def alloc_buffer(self, shape, dtype, hostbuf=None):
        size = int(dtype().itemsize * prod(shape))
        flags = pyopencl.mem_flags.READ_WRITE
        if hostbuf is not None:
            flags |= pyopencl.mem_flags.COPY_HOST_PTR
        return pyopencl.Buffer(self.ctx, flags, size, hostbuf=hostbuf)

    def alloc_local_memory(self, size):
        return pyopencl.LocalMemory(size)

cl = CLContext()

class KernelCounter:
    cnt = defaultdict(int)

def broadcast(a, b):
    # rule: https://numpy.org/doc/stable/user/basics.broadcasting.html
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
        ret = a.__class__(shape=a.shape, dtype=a.dtype)  # TODO
    assert ret.c_contiguous, f"ret must be contiguous. {ret}"
    code_map = {"noop": "a", "neg": "-a", "log": "log(a)", "exp": "exp(a)", "relu": "max(a, 0.0f)", "sign": "sign(a)"}
    op = cl.build("unary_op", f"""__kernel void unary_op(
        {''.join([f'int a_s{i}, int res_s{i}, ' for i in range(a.ndim)])}
        int a_ofst, __global const float *A, __global float *B) {{
      int a_i=0, idx=0, gl_id=get_global_id(0); int ptr=gl_id;
      {''.join([f'idx=ptr/res_s{i}; ptr%=res_s{i}; a_i+=idx*a_s{i};' for i in range(a.ndim)])}
      float a=A[a_i+a_ofst];
      B[gl_id]={code_map[name]};
    }}""")
    args = [int32(s) for ss in zip(a.strides, ret.strides) for s in ss] + [int32(a.offset)]
    e = op((prod(a.shape),), None, *args, a.buffer, ret.buffer)
    if GRAPH: e.wait()
    KernelCounter.cnt["unary"] += 1
    return ret

def binary_op(name, a, b, ret=None):
    a, b = broadcast(a, b)
    if ret is None:
        ret = a.__class__(shape=a.shape, dtype=a.dtype)
    assert ret.c_contiguous, f"ret must be contiguous. {ret}"
    ret_strides = (1,) if not ret.strides else ret.strides
    code_map = {"add": "a+b", "sub": "a-b", "truediv": "a/b", "mul": "a*b", "pow": "pow(a,b)", "eq": "(float)isequal(a,b)", "gt": "(float)isgreater(a,b)", "ge": "(float)isgreaterequal(a,b)", "drelu": "b>0?a:0.0f"}
    op = cl.build("binary_op", f"""__kernel void binary_op(
        {''.join([f'int a_s{i}, int b_s{i}, int res_s{i}, ' for i in range(a.ndim)])}
        int a_ofst, int b_ofst, __global const float *A, __global const float *B, __global float *C) {{
      int a_i=0, b_i=0, idx=0, gl_id=get_global_id(0); int ptr=gl_id;
      {''.join(f'idx=ptr/res_s{i}; ptr%=res_s{i}; a_i+=idx*a_s{i}; b_i+=idx*b_s{i};' for i in range(a.ndim))}
      float a=A[a_i+a_ofst], b=B[b_i+b_ofst];
      C[gl_id] = {code_map[name]};
    }}""")
    args = [int32(s) for ss in zip(a.strides, b.strides, ret_strides) for s in ss]
    args += [int32(s) for s in [a.offset, b.offset]]
    e = op((prod(a.shape),), None, *args, a.buffer, b.buffer, ret.buffer)
    if GRAPH: e.wait()
    KernelCounter.cnt["binary"] += 1
    return ret

def matmul_op(a, b):
    # rule: https://numpy.org/doc/stable/reference/generated/numpy.matmul.html
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
    assert ret.c_contiguous, f"ret must be contiguous. {ret}"
    BS, M, K, N = prod(a.shape[:-2]), a.shape[-2], a.shape[-1], b.shape[-1]
    gs = 1
    while gs <= 8 and M % gs == 0 and N % gs == 0 and K % gs == 0 and gs <= K and gs <= M and gs <= N:
        gs *= 2
    gs //= 2
    if DEBUG>1: print(f"[DEBUG] BS:{BS} M:{M} K:{K} N:{N} grp_size:{gs}")
    src = f"""#define GS {gs}
    __kernel void matmul_op(int BS, int M, int N, int K,
        {''.join(f'int A_s{i}, int B_s{i}, ' for i in range(3))} int a_ofst, int b_ofst,
        __global const float *A, __global const float *B, __global float *C) {{
      int bs=get_global_id(0), m=get_global_id(1), n=get_global_id(2), i=get_local_id(1), j=get_local_id(2);
      __local float Alcl[GS][GS], Blcl[GS][GS];
      float acc = 0.0f;
      for (int t=0; t<K/GS; t++) {{
        Alcl[i][j] = A[bs*A_s0+m*A_s1+(t*GS+j)*A_s2+a_ofst];
        Blcl[i][j] = B[bs*B_s0+(t*GS+i)*B_s1+n*B_s2+b_ofst];
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int k=0; k<GS; k++) acc += Alcl[i][k] * Blcl[k][j];
        barrier(CLK_LOCAL_MEM_FENCE);
      }}
      C[bs*M*N+m*N+n] = acc;
    }}"""
    op = cl.build("matmul_op", src)
    strides = [s for ss in zip(a.strides, b.strides) for s in ss]
    args = [int32(x) for x in [BS, M, N, K] + strides + [a.offset, b.offset]]
    e = op((BS, M, N), (1, gs, gs), *args, a.buffer, b.buffer, ret.buffer)
    if GRAPH: e.wait()
    KernelCounter.cnt["matmul"] += 1
    for axis in squeezes:
        ret = ret.squeeze(axis)
    return ret

def contiguous_op(x):
    ret = x.__class__(shape=x.shape, dtype=x.dtype)
    x_ndim, x_shape, x_strides = x.ndim, x.shape, x.strides
    if not x_ndim:
        x_ndim, x_shape, x_strides = 1, (1,), (1,)
    def_args = "".join([f"int a{i}, int b{i}, " for i in range(x_ndim)])
    def_args += "int ofst"
    def_strides = "".join([f"int _s{i}="+"*".join(f"a{j}" for j in range(i+1, x_ndim)) + ";" for i in range(x_ndim-1)])
    def_strides += f"int _s{x_ndim-1}=1;"
    def_indices = "".join(f"int _i{i}=curr/_s{i}; curr%=_s{i}; " for i in range(x_ndim))
    addr = "+".join([f"b{i}*_i{i}" for i in range(x_ndim)])
    src = f"""__kernel void contiguous_op({def_args}, __global const float *A, __global float *B) {{
      int curr = get_global_id(0);
      {def_strides} {def_indices}
      B[get_global_id(0)] = A[{addr}+ofst];
    }}"""
    op = cl.build("contiguous_op", src)
    args = [int32(s) for ss in zip(x_shape, x_strides) for s in ss]
    e = op((prod(x_shape),), None, *args, int32(x.offset), x.buffer, ret.buffer)
    if GRAPH: e.wait()
    KernelCounter.cnt["contig"] += 1
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
    max_work_group_size = cl.queue.device.max_work_group_size
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
    op = cl.build("reduce_op", f"""__kernel void reduce_op(int size, int ofst,
        __global const float *A, __local float *B, __global float *C) {{
      {''.join([f'int gl_id_{i}=get_global_id({i});int gl_s_{i}=get_global_size({i});int grp_id_{i}=get_group_id({i});int grp_s_{i}=get_local_size({i});' for i in range(ndim)])}
      int lcl_id=get_local_id({axis});
      B[lcl_id] = gl_id_{axis}<size?A[{gl2lcl}-{offset}+ofst]:{padval_map[name]};
      barrier(CLK_LOCAL_MEM_FENCE);
      for (int stride=grp_s_{axis}>>1; stride>0; stride>>=1) {{
        float a = B[lcl_id], b = B[lcl_id+stride];
        if (lcl_id<stride) B[lcl_id]={code_map[name]};
        barrier(CLK_LOCAL_MEM_FENCE);
      }}
      if (lcl_id == 0) C[{lcl2gl}]=B[0];
    }}""")
    local_mem = cl.alloc_local_memory(x.dtype().itemsize * grp_size)
    local_size = tuple(grp_size if i == axis else 1 for i in range(ndim))
    e = op(global_size, local_size, int32(size), int32(x.offset), x.buffer, local_mem, ret.buffer)
    if GRAPH: e.wait()
    KernelCounter.cnt["reduce"] += 1
    if DEBUG>1: print(f"[DEBUG] x_shp: {x_shp} ret_shape: {ret_shape} grp_size: {grp_size} n_grps: {n_grps} size: {size} global_size: {global_size} local_size: {local_size} axis={axis} ndim={ndim} offset={offset}")
    if n_grps > 1:
        ret = reduce_op(name, ret, axis=axis, keepdims=keepdims)  # recursive reduce (inefficient)
    return ret


class CLArray(Array):
    # https://numpy.org/doc/stable/dev/internals.html#numpy-internals
    def __init__(self, data=None, shape=None, dtype=float32):
        if isinstance(data, pyopencl.Buffer):
            self.buffer = data
            assert shape is not None, "cannot infer shape when initialize using clbuffer"
        else:
            if data is not None:
                data = np.asarray(data, dtype=dtype)
                shape = data.shape
            else:
                assert shape is not None, "cannot infer shape when without data"
            self.buffer = cl.alloc_buffer(shape, dtype, data)
        self.shape, self.dtype = tuple(shape), dtype
        self.strides = tuple(prod(shape[i+1:]) for i in range(len(shape)))
        self.offset = 0  # offset relative to the beginning of the buffer
        self.c_contiguous, self.f_contiguous = True, False
        self.update_contiguousness()
        self.register_ops()

    @staticmethod
    def as_gpu_array(obj):
        if not isinstance(obj, CLArray):
            obj = CLArray(obj)
        return obj

    @property
    def size(self):
        return self.buffer.size

    @property
    def ndim(self):
        return len(self.shape)

    def __repr__(self):
        return (f"<CLArray dtype={self.dtype} shape={self.shape} strides={self.strides} size={self.size} contiguous={self.c_contiguous}>")

    def register_ops(self):
        cls = self.__class__
        for op in ("add", "sub", "mul", "truediv", "pow"):
            setattr(cls, f"__{op}__",
                    (lambda op: lambda a, b: binary_op(op, a, self.as_gpu_array(b)))(op))
            setattr(cls, f"__i{op}__",
                    (lambda op: lambda a, b: binary_op(op, a, self.as_gpu_array(b), ret=a))(op))
            setattr(cls, f"__r{op}__",
                    (lambda op: lambda a, b: binary_op(op, self.as_gpu_array(b), a))(op))
        for op in ("eq", "ge", "gt"):
            setattr(cls, f"__{op}__",
                    (lambda op: lambda a, b: binary_op(op, a, self.as_gpu_array(b)))(op))
        setattr(cls, f"__matmul__", lambda a, b: matmul_op(a, self.as_gpu_array(b)))
        setattr(cls, f"__neg__", lambda a: unary_op("neg", a))

    def sum(self, axis=None, keepdims=False):
        arr = self.contiguous() if not self.c_contiguous else self
        return reduce_op("sum", arr, axis=axis, keepdims=keepdims)

    def max(self, axis=None, keepdims=False):
        arr = self.contiguous() if not self.c_contiguous else self
        return reduce_op("max", arr, axis=axis, keepdims=keepdims)

    def relu(self, inplace=False):
        return unary_op("relu", self, ret=self if inplace else None)

    def exp(self):
        return unary_op("exp", self)

    def log(self):
        return unary_op("log", self)

    def drelu(self, other):
        return binary_op("drelu", self, self.as_gpu_array(other))

    def __getitem__(self, key):
        # TODO: handle step
        is_basic = lambda k: isinstance(k, (slice, int))
        assert is_basic(key) or all(is_basic(k) for k in key), \
                f"Advantage indexing not supported yet. {key}"
        key = (key,) if is_basic(key) else key
        inst = copy.copy(self)
        reduce = []
        shape = list(inst.shape)
        for i, k in enumerate(key):
            if isinstance(k, int):  # indexing
                if k < 0: k += inst.shape[i]
                assert 0 <= k < inst.shape[i], f"Invalid indexing {key[i]} for tensor {inst.shape}"
                inst.offset += inst.strides[i] * k
                reduce.append(i)
            if isinstance(k, slice):  # slicing
                start = 0 if k.start is None else k.start
                if start < 0: start += inst.shape[i]
                stop = inst.shape[i] if k.stop is None else k.stop
                if stop < 0: stop += inst.shape[i]
                assert 0 <= start < stop <= inst.shape[i], f"Invalid slicing {key[i]} for tensor {inst.shape}"
                shape[i] = stop - start
                inst.offset += inst.strides[i] * start
                inst.c_contiguous, inst.f_contiguous = False, False  # TODO: is still contiguous under certain conditions
        inst.shape = tuple(s for i, s in enumerate(shape) if i not in reduce)
        inst.strides = tuple(s for i, s in enumerate(inst.strides) if i not in reduce)
        return inst

    def __setitem__(self, key, value):
        item = self[key]
        # unary_op("noop", value, ret=item)
        assert False, "TODO: implement assign ops"

    @classmethod
    def empty(cls, shape, dtype=float32):
        return cls(shape=shape, dtype=dtype)

    @classmethod
    def zeros(cls, shape, dtype=float32):
        return cls(shape=shape, dtype=dtype).fill(0)

    @classmethod
    def ones(cls, shape, dtype=float32):
        return cls(shape=shape, dtype=dtype).fill(1)

    @classmethod
    def full(cls, shape, value, dtype=float32):
        return cls(shape=shape, dtype=dtype).fill(value)

    @classmethod
    def uniform(cls, a, b, shape, dtype=float32):
        buffer = cl.rng.uniform(a=a, b=b, shape=shape, dtype=dtype, cq=cl.queue).data
        return cls(data=buffer, shape=shape, dtype=dtype)

    @classmethod
    def normal(cls, loc, scale, shape, dtype=float32):
        buffer = cl.rng.normal(mu=loc, sigma=scale, shape=shape, dtype=dtype, cq=cl.queue).data
        return cls(data=buffer, shape=shape, dtype=dtype)

    def numpy(self):
        data = np.empty(self.shape, dtype=self.dtype)
        pyopencl.enqueue_copy(cl.queue, data, self.contiguous().buffer, is_blocking=True)
        return data

    def contiguous(self):
        return contiguous_op(self)

    def reshape(self, shape):
        if -1 in shape:
            size = prod(self.shape)
            assert shape.count(-1) <= 1, "Only one dimension can be inferred"
            axis = shape.index(-1)
            infer = prod([s for s in shape if s != -1])
            assert size % infer == 0, f"Shape {shape} invalid for size {size}"
            shape = (*shape[:axis], size // infer, *shape[axis+1:])

        assert prod(shape) == prod(self.shape), f"Can not reshape {self.shape} to {shape}"
        if self.c_contiguous or self.f_contiguous:
            inst = copy.copy(self)
            if self.c_contiguous:
                strides = (prod(shape[i+1:]) for i in range(len(shape)))
            else:
                strides = (prod(shape[:i]) for i in range(len(shape)))
            inst.shape, inst.strides = tuple(shape), tuple(strides)
            inst.update_contiguousness()
        else:
            inst = self.contiguous().reshape(shape)
        return inst

    def expand(self, shape):
        inst = copy.copy(self)
        assert len(shape) == inst.ndim
        strides = []
        for i, (s1, s2) in enumerate(zip(inst.shape, shape)):
            if s1 < s2:
                assert s1 == 1
            strides.append(0 if s1 < s2 else inst.strides[i])
        inst.shape, inst.strides = tuple(shape), tuple(strides)
        inst.c_contiguous, inst.f_contiguous = False, False
        return inst

    def squeeze(self, axis=None):
        if axis is None:
            axis = [i for i, s in enumerate(self.shape) if s == 1]
        elif isinstance(axis, int):
            axis = [axis]
        assert isinstance(axis, (list, tuple))
        axis = [a if a != -1 else self.ndim - 1 for a in axis]
        shape = [s for i, s in enumerate(self.shape) if i not in axis or self.shape[i] != 1]
        if shape == self.shape:
            return self
        return self.reshape(shape)

    def storage(self):
        data = np.empty((self.buffer.size // self.dtype().itemsize,), dtype=self.dtype)
        pyopencl.enqueue_copy(cl.queue, data, self.buffer, is_blocking=True)
        return data

    def fill(self, value):
        pyopencl.enqueue_fill_buffer(cl.queue, self.buffer, self.dtype(value), 0, self.size)
        return self

    def permute(self, axes):
        inst = copy.copy(self)
        inst.strides = tuple(inst.strides[a] for a in axes)
        inst.shape = tuple(inst.shape[a] for a in axes)
        inst.update_contiguousness()
        return inst

    def update_contiguousness(self):
        strides = [self.strides[i] for i in range(self.ndim) if self.shape[i] != 1]
        sorted_strides = sorted(strides)
        self.f_contiguous = sorted_strides == strides
        self.c_contiguous = sorted_strides[::-1] == strides

