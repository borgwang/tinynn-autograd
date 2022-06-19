import os
import sys
import numpy as np

from core.tensor import Tensor
os.environ["PYOPENCL_COMPILER_OUTPUT"] = "1"

np.random.seed(0)

data1 = 1 * np.ones((3, 3), dtype=np.float32)
data2 = 4 * np.ones((3, 3), dtype=np.float32)
print("data1: \n", data1)
print("data2: \n", data2)
print("CPU result: \n", data1 + data2)
#print("CPU result: \n", np.matmul(data1, data2))

x1 = Tensor(data1, requires_grad=True, dtype=np.float32).gpu()
x2 = Tensor(data2, requires_grad=True, dtype=np.float32).gpu()
y = x1 + x2
#y = x1 @ x2
print("GPU result: \n", y.cpu())
import pdb; pdb.set_trace()

x1.zero_grad()
x2.zero_grad()
y.backward()

print(x1.grad.cpu())
print(x2.grad.cpu())

import pdb; pdb.set_trace()


params = {
    "w": Tensor(np.random.normal(0, 1.0, (3, 3)), requires_grad=True).gpu(),
    "b": Tensor(np.random.normal(0, 1.0, 3).gpu(), requires_grad=True).gpu()
}

learng_rate = 3e-4
loss_list = []
for e in range(101):
    # set gradient to zero
    for param in params.values():
        param.zero_grad()

    # forward
    predicted = x @ params["w"] + params["b"]
    err = predicted - y
    loss = (err * err).sum()

    # backward automatically
    loss.backward()

    # updata parameters (gradient descent)
    for param in params.values():
        param -= learng_rate * param.grad

    loss_list.append(loss.values)
    if e % 10 == 0:
        print("epoch-%i \tloss: %.4f" % (e, loss.values))
