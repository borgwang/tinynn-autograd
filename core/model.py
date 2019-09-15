"""Model class manage the network, loss function and optimizer."""

import pickle


class Model(object):

    def __init__(self, net, loss, optimizer):
        self.net = net
        self.loss = loss
        self.optimizer = optimizer

        self._phase = "TRAIN"

    def forward(self, inputs):
        return self.net.forward(inputs)

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.net, f, -1)
        print("Model saved in %s." % path)

    def load(self, path):
        # compatibility checking
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

    def get_phase(self):
        return self._phase

    def set_phase(self, phase):
        assert phase in ("TRAIN", "TEST")
        self.net.set_phase(phase)
        self._phase = phase

    def step(self):
        # grad all grads
        all_grads = []
        params = self.net.get_parameters()
        for param in params:
            grad = dict()
            for k, v in param.items():
                grad[k] = param[k].grad
            all_grads.append(grad)

        # compute step
        steps = self.optimizer.compute_step(all_grads, params)

        # apply grad
        for step, param in zip(steps, params):
            for k, v in param.items():
                param[k] += step[k]

    def zero_grad(self):
        params = self.net.get_parameters()
        for param in params:
            for p in param.values():
                if p is not None:
                    p.zero_grad()
