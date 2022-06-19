import numpy as np

from core.tensor import Tensor

x = Tensor(np.random.normal(0, 1, (100, 128))).gpu()
y = Tensor(np.random.normal(0, 1, (100, 2))).gpu()
w = Tensor(np.random.normal(0, 1, (128, 2)), requires_grad=True).gpu()

for epoch in range(500):
    # zero gradients
    w.zero_grad()
    # forward
    pred = x @ w
    err = pred - y
    loss = (err * err).sum()
    # backward
    loss.backward()
    # updata parameters
    w -= 0.001 * w.grad

    if epoch % 10 == 0:
        print(f"epoch-{epoch}  loss: {loss.values.get():.3f}")
