"""Loss functions"""

import numpy as np

import core.ops as ops


class BaseLoss(object):

    def loss(self, predicted, actual):
        raise NotImplementedError


class SoftmaxCrossEntropyLoss(BaseLoss):

    def __init__(self, weight=None):
        """
        L = weight[class] * (-log(exp(x[class]) / sum(exp(x))))
        :param weight: A 1D tensor [n_classes] assigning weight to each corresponding sample.
        """
        weight = np.asarray(weight) if weight is not None else weight
        self._weight = weight

    def loss(self, logits, labels):
        m = logits.shape[0]
        exps = ops.exp(logits - logits.max())
        p = exps / exps.sum()
        nll = -ops.log((p * labels).sum(1))

        if self._weight is not None:
            nll *= self._weight[labels]
        return nll.sum() / m
