import numpy as np


class Loss:
    def loss(self, predicted, actual):
        raise NotImplementedError

class SoftmaxCrossEntropyLoss(Loss):
    def __init__(self, weight=None):
        """
        L = weight[class] * (-log(exp(x[class]) / sum(exp(x))))
        :param weight: A 1D tensor [n_classes] assigning weight to each corresponding sample.
        """
        self._weight = np.asarray(weight) if weight is not None else weight

    def loss(self, logits, labels):
        m = logits.shape[0]
        exps = (logits - logits.max(axis=1, keepdims=True)).exp()
        exps = logits.exp()
        p = exps / exps.sum()
        nll = -(p * labels).sum(axis=1)

        if self._weight is not None:
            nll *= self._weight[labels]
        return nll.sum() / m

class SigmoidCrossEntropy(Loss):

    def loss(self, logits, labels):
        p = 1.0 / (1.0 + (-logits).exp())
        cost = logits * (1 - labels) - p.log()
        return cost.sum() / labels.shape[0]
