"""Various optimization algorithms and learning rate schedulers."""

import numpy as np


class BaseOptimizer(object):

    def __init__(self, lr, weight_decay):
        self.lr = lr
        self.weight_decay = weight_decay

    def compute_step(self, grads, params):
        # flatten all gradients into 1-dim array
        flatten_grads = np.concatenate(
            [np.ravel(v) for grad in grads for v in grad.values()])
        # compute step according to derived class method
        flatten_step = self._compute_step(flatten_grads)

        p = 0 # linear block pointer
        steps = list() # all layer of steps in restored shape
        for param in params:
            layer = dict() # one layer of steps in restored shape
            for k, v in param.items():
                # the number of elements in v
                block = np.prod(v.shape)
                # restore the shape for a block of flatten_step
                _step = flatten_step[p:p+block].reshape(v.shape)
                # apply weight_decay if specified
                # _step -= self.weight_decay * v
                # set the restored step to parameter key
                layer[k] = _step
                # count the block
                p += block
            steps.append(layer)
        return steps

    def _compute_step(self, grad):
        raise NotImplementedError


class SGD(BaseOptimizer):

    def __init__(self, lr, weight_decay=0.0):
        super().__init__(lr, weight_decay)

    def _compute_step(self, grad):
        return - self.lr * grad


class Adam(BaseOptimizer):

    def __init__(self,
                 lr=0.001,
                 beta1=0.9,
                 beta2=0.999,
                 epsilon=1e-8,
                 weight_decay=0.0):
        super().__init__(lr, weight_decay)
        self._b1 = beta1
        self._b2 = beta2
        self._eps = epsilon

        self._t = 0
        self._m = 0
        self._v = 0

    def _compute_step(self, grad):
        self._t += 1

        self._m += (1.0 - self._b1) * (grad - self._m)
        self._v += (1.0 - self._b2) * (grad ** 2 - self._v)

        # bias correction
        _m = self._m / (1 - self._b1 ** self._t)
        _v = self._v / (1 - self._b2 ** self._t)

        step = -self.lr * _m / (_v ** 0.5 + self._eps)

        return step


class RMSProp(BaseOptimizer):
    """
    RMSProp maintain a moving (discounted) average of the square of gradients.
    Then divide gradients by the root of this average.

    mean_square = decay * mean_square{t-1} + (1-decay) * grad_t**2
    mom = momentum * mom{t-1} + lr * grad_t / sqrt(mean_square + epsilon)
    """
    def __init__(self,
                 lr=0.01,
                 decay=0.99,
                 momentum=0.0,
                 epsilon=1e-8,
                 weight_decay=0.0):
        super().__init__(lr, weight_decay)
        self._decay = decay
        self._momentum = momentum
        self._eps = epsilon

        self._ms = 0
        self._mom = 0

    def _compute_step(self, grad):
        self._ms += (1 - self._decay) * (grad ** 2 - self._ms)
        self._mom = self._momentum * self._mom + \
            self.lr * grad / (self._ms + self._eps) ** 0.5

        step = -self._mom
        return step


class Momentum(BaseOptimizer):
    """
     accumulation = momentum * accumulation + gradient
     variable -= learning_rate * accumulation
    """
    def __init__(self, lr, momentum=0.9, weight_decay=0.0):
        super().__init__(lr, weight_decay)
        self._momentum = momentum
        self._acc = 0

    def _compute_step(self, grad):
        self._acc = self._momentum * self._acc + grad
        step = -self.lr * self._acc
        return step


class Adagrad(BaseOptimizer):
    """
    AdaGrad optimizer (http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf)
    accumulation = - (learning_rate / sqrt(G + epsilon)) * gradient
    where G is the element-wise sum of square gradient
    """
    def __init__(self, lr, weight_decay=0.0, epsilon=1e-8):
        super().__init__(lr, weight_decay)
        self._G = 0
        self._eps = epsilon

    def _compute_step(self, grad):
        self._G += grad ** 2
        adjust_lr = self.lr / (self._G + self._eps) ** 0.5
        step = -adjust_lr * grad
        return step


class Adadelta(BaseOptimizer):
    """
    Adadelta algorithm (https://arxiv.org/abs/1212.5701)
    """
    def __init__(self, lr=1.0, weight_decay=0.0, decay=0.9, epsilon=1e-8):
        super().__init__(lr, weight_decay)
        self._eps = epsilon
        self._decay = decay
        self._Eg = 0  # running average of square gradient
        self._delta = 0  # running average of delta

    def _compute_step(self, grad):
        self._Eg += (1 - self._decay) * (grad ** 2 - self._Eg)
        std = (self._delta + self._eps) ** 0.5
        delta = grad * (std / (self._Eg + self._eps) ** 0.5)
        step = - self.lr * delta
        self._delta += (1 - self._decay) * (delta ** 2 - self._delta)
        return step
