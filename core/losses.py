import numpy as np


class Loss:
    def loss(self, predicted, actual):
        raise NotImplementedError

class SoftmaxCrossEntropyLoss(Loss):

    def loss(self, logits, labels):
        m = logits.shape[0]
        #exps = (logits - logits.max(axis=1, keepdims=True)).exp()
        exps = logits.exp()
        expsum = exps.sum()
        p = exps / expsum
        l = (p * labels).sum(axis=1)
        nll = -l.log()
        ret = nll.sum() / m
        return ret


class SigmoidCrossEntropy(Loss):

    def loss(self, logits, labels):
        p = 1.0 / (1.0 + (-logits).exp())
        cost = logits * (1 - labels) - p.log()
        return cost.sum() / labels.shape[0]
