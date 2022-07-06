import numpy as np


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

