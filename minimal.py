import os
import sys
import numpy as np

from core.tensor import Tensor
os.environ["PYOPENCL_COMPILER_OUTPUT"] = "1"

np.random.seed(0)

"""
# TODO: unit test!
data1 = 1 * np.ones((10, 3), dtype=np.float32)
data2 = 4 * np.ones((3, 4), dtype=np.float32)
print("data1: \n", data1)
print("data2: \n", data2)
print("CPU result: \n", data1 @ data2)
#print("CPU result: \n", np.matmul(data1, data2))

x1 = Tensor(data1, requires_grad=True, dtype=np.float32).gpu()
x2 = Tensor(data2, requires_grad=True, dtype=np.float32).gpu()
y = x1 @ x2
print("GPU result: \n", y.cpu().values)

x1.zero_grad()
x2.zero_grad()
y.backward()

print(x1.grad.cpu())
print(x2.grad.cpu())
"""

"""
from core.gpu_ops import contiguous_transpose_op
data = np.arange(15, dtype=np.float32).reshape((3, 5))
x = Tensor(data).gpu()
y = contiguous_transpose_op(x)
print(data)
print(y.get().reshape((5, 3)).astype(int))
import pdb; pdb.set_trace()
"""


bs = 32
odim = 8
hdim = 2
x = Tensor(np.random.normal(0, 1, (bs, hdim)), requires_grad=True, dtype=np.float32).gpu()
y = Tensor(np.random.normal(0, 1, (bs, odim)), requires_grad=True, dtype=np.float32).gpu()

params = {
    "w": Tensor(np.random.normal(0, 1.0, (hdim, odim)), requires_grad=True).gpu(),
    #"b": Tensor(np.random.normal(0, 1.0, (32, 2)), requires_grad=True).gpu()
}

learing_rate = 3e-4
loss_list = []
for e in range(1000):
    # set gradient to zero
    for param in params.values():
        param.zero_grad()

    # forward
    #pred = x @ params["w"] + params["b"]
    pred = x @ params["w"]
    err = pred - y
    loss = (err * err).sum()

    # backward automatically
    loss.backward()

    # updata parameters (gradient descent)
    for param in params.values():
        param -= learing_rate * param.grad

    loss_list.append(loss.values)
    if e % 1 == 0:
        print("epoch-%i \tloss: %.4f" % (e, float(loss.values.get())))
