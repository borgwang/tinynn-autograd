import numpy as np

class Evaluator:
    @classmethod
    def evaluate(cls, predictions, targets):
        raise NotImplementedError("Must specify evaluator.")

class AccEvaluator(Evaluator):
    @classmethod
    def evaluate(cls, predictions, targets):
        total_num = len(predictions)
        hit_num = int(np.sum(predictions == targets))
        return {"total_num": total_num,
                "hit_num": hit_num,
                "accuracy": 1.0 * hit_num / total_num}

class MSEEvaluator(Evaluator):
    @classmethod
    def evaluate(cls, predictions, targets):
        assert predictions.shape == targets.shape
        if predictions.ndim == 1:
            mse = np.mean(np.square(predictions - targets))
        elif predictions.ndim == 2:
            mse = np.mean(np.sum(np.square(predictions - targets), axis=1))
        else:
            raise ValueError("predision supposes to have 1 or 2 dim.")
        return {"mse": mse}

class MAEEvaluator(Evaluator):
    @classmethod
    def evaluate(cls, predictions, targets):
        assert predictions.shape == targets.shape
        if predictions.ndim == 1:
            mse = np.mean(np.abs(predictions - targets))
        elif predictions.ndim == 2:
            mse = np.mean(np.sum(np.abs(predictions - targets), axis=1))
        else:
            raise ValueError("predision supposes to have 1 or 2 dim.")
        return {"mse": mse}

