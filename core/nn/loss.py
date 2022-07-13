class Loss:
    def __call__(self, predicted, actual):
        raise NotImplementedError

class SoftmaxCrossEntropyLoss(Loss):
    def __call__(self, logits, labels):
        m = logits.shape[0]
        exps = (logits - logits.max(axis=1, keepdims=True)).exp()
        p = exps / exps.sum()
        liklihood = (p * labels).sum(axis=1)
        nll = -liklihood.log()
        return nll.sum() / m

