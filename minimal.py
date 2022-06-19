import os
import sys
import numpy as np

from core.tensor import Tensor
os.environ["PYOPENCL_COMPILER_OUTPUT"] = "1"

np.random.seed(0)

bs = 32
odim = 8
hdim = 2
x = Tensor(np.random.normal(0, 1, (bs, hdim)), requires_grad=True, dtype=np.float32).gpu()
y = Tensor(np.random.normal(0, 1, (bs, odim)), requires_grad=True, dtype=np.float32).gpu()

params = {
    "w": Tensor(np.random.normal(0, 1.0, (hdim, odim)), requires_grad=True).gpu(),
    "b": Tensor(np.random.normal(0, 1.0, (odim,)), requires_grad=True).gpu()
}

learing_rate = 3e-4
loss_list = []
for e in range(1000):
    # set gradient to zero
    for param in params.values():
        param.zero_grad()

    # forward
    pred = x @ params["w"] * params["b"]
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
