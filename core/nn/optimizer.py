from collections import defaultdict

class Optimizer:
    def __init__(self, params, lr, weight_decay):
        self.lr = lr
        self.weight_decay = weight_decay
        self.t = 0
        self.params = params

    def step(self):
        self.t += 1
        for i, param in enumerate(self.params):
            for name in param:
                param[name].values += self._get_step(param[name].grad, key=f"{i}-{name}")

    def _get_step(self, grad):
        raise NotImplementedError

class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.9, weight_decay=0.0):
        super().__init__(params, lr, weight_decay)
        self._momentum = momentum
        self._acc = defaultdict(int)

    def _get_step(self, grad, key):
        self._acc[key] = self._momentum * self._acc[key] + grad
        return -self.lr * self._acc[key]

class RMSProp(Optimizer):
    def __init__(self, params, lr=0.01, decay=0.99, momentum=0.0, epsilon=1e-8, weight_decay=0.0):
        super().__init__(params, lr, weight_decay)
        self._rho = decay
        self._momentum = momentum
        self._epsilon = epsilon
        self._rms, self._mom = defaultdict(int), defaultdict(int)

    def _get_step(self, grad, key):
        self._rms[key] += (1 - self._rho) * (grad ** 2 - self._rms[key])
        self._mom[key] = self._momentum * self._mom[key] + self.lr * grad / \
                (self._rms[key] + self._epsilon)**0.5
        return -self._mom[key]

class Adam(Optimizer):
    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.0):
        super().__init__(params, lr, weight_decay)
        self._b1, self._b2, self._epsilon = beta1, beta2, epsilon
        self._m, self._v = defaultdict(int), defaultdict(int)

    def _get_step(self, grad, key):
        self._m[key] += (1.0 - self._b1) * (grad - self._m[key])
        self._v[key] += (1.0 - self._b2) * (grad ** 2 - self._v[key])
        # bias correction
        _m = self._m[key] / (1 - self._b1 ** self.t)
        _v = self._v[key] / (1 - self._b2 ** self.t)
        return -self.lr * _m / (_v ** 0.5 + self._epsilon)

