from collections import defaultdict
import numpy as np
import pickle

class Model:
    def __init__(self, net, loss, optimizer):
        self.net = net
        self.loss = loss
        self.optimizer = optimizer

    def forward(self, inputs):
        return self.net.forward(inputs)

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.net, f, -1)
        print("Model saved in %s." % path)

    def load(self, path):
        with open(path, "rb") as f:
            net = pickle.load(f)
        for l1, l2 in zip(self.net.layers, net.layers):
            if l1.shape != l2.shape:
                raise ValueError("Incompatible architecture. %s in loaded model"
                                 " and %s in defined model." %
                                 (l1.shape, l2.shape))
            else:
                print("%s: %s" % (l1.name, l1.shape))
        self.net = net
        print("Restored model from %s." % path)

    def step(self):
        all_grads = []
        params = self.net.get_parameters()
        for param in params:
            grad = dict()
            for k, v in param.items():
                grad[k] = param[k].grad
            all_grads.append(grad)
        self.optimizer.compute_step(all_grads, params)

    def zero_grad(self):
        params = self.net.get_parameters()
        for param in params:
            for p in param.values():
                if p is not None:
                    p.zero_grad()

class Net:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, inputs):
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    def get_parameters(self):
        return [layer.params for layer in self.layers]

    def set_parameters(self, params):
        for i, layer in enumerate(self.layers):
            assert layer.params.keys() == params[i].keys()
            for key in layer.params.keys():
                assert layer.params[key].shape == params[i][key].shape
                layer.params[key] = params[i][key]

    def to(self, device):
        for layer in self.layers:
            layer.to(device)
        return self


class Loss:
    def loss(self, predicted, actual):
        raise NotImplementedError

class SoftmaxCrossEntropyLoss(Loss):

    def loss(self, logits, labels):
        m = logits.shape[0]
        exps = (logits - logits.max(axis=1, keepdims=True)).exp()
        expsum = exps.sum()
        p = exps / expsum
        l = (p * labels).sum(axis=1)
        nll = -l.log()
        ret = nll.sum() / m
        return ret

class Optimizer:
    def __init__(self, lr, weight_decay):
        self.lr = lr
        self.weight_decay = weight_decay
        self.t = 0

    def compute_step(self, grads, params):
        self.t += 1
        for i, (grad, param) in enumerate(zip(grads, params)):
            for k, v in grad.items():
                key = f"{i}-{k}"
                param[k].values += self._compute_step(v, key)

    def _compute_step(self, grad):
        raise NotImplementedError

class SGD(Optimizer):
    def __init__(self, lr, momentum=0.9, weight_decay=0.0):
        super().__init__(lr, weight_decay)
        self._momentum = momentum
        self._acc = defaultdict(int)

    def _compute_step(self, grad, key):
        self._acc[key] = self._momentum * self._acc[key] + grad
        return -self.lr * self._acc[key]

class RMSProp(Optimizer):
    def __init__(self, lr=0.01, decay=0.99, momentum=0.0, epsilon=1e-8, weight_decay=0.0):
        super().__init__(lr, weight_decay)
        self._rho = decay
        self._momentum = momentum
        self._epsilon = epsilon
        self._rms, self._mom = defaultdict(int), defaultdict(int)

    def _compute_step(self, grad, key):
        self._rms[key] += (1 - self._rho) * (grad ** 2 - self._rms[key])
        self._mom[key] = self._momentum * self._mom[key] + self.lr * grad / \
                (self._rms[key] + self._epsilon)**0.5
        return -self._mom[key]

class Adam(Optimizer):
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.0):
        super().__init__(lr, weight_decay)
        self._b1, self._b2, self._epsilon = beta1, beta2, epsilon
        self._m, self._v = defaultdict(int), defaultdict(int)

    def _compute_step(self, grad, key):
        self._m[key] += (1.0 - self._b1) * (grad - self._m[key])
        self._v[key] += (1.0 - self._b2) * (grad ** 2 - self._v[key])
        # bias correction
        _m = self._m[key] / (1 - self._b1 ** self.t)
        _v = self._v[key] / (1 - self._b2 ** self.t)
        return -self.lr * _m / (_v ** 0.5 + self._epsilon)

