import numpy as np


class Loss:
    def loss(self, predicted, actual):
        raise NotImplementedError

class SoftmaxCrossEntropyLoss(Loss):

    def loss(self, logits, labels):
        m = logits.shape[0]
        #exps = (logits - logits.max(axis=1, keepdims=True)).exp()
        exps = logits.exp()
        p = exps / exps.sum()
        nll = -((p * labels).sum(axis=1).log())
        return nll.sum() / m


class SigmoidCrossEntropy(Loss):

    def loss(self, logits, labels):
        p = 1.0 / (1.0 + (-logits).exp())
        cost = logits * (1 - labels) - p.log()
        return cost.sum() / labels.shape[0]
