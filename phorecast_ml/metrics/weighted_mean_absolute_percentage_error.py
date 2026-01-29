import keras
from keras import backend as K
from keras import ops


@keras.saving.register_keras_serializable(package="Metrics", name="wMAPE")
def sum_difference_metric(y_true, y_pred):
    """
    This metric is used to calculate the wMAPE
    :param y_true: ground truth tensor
    :param y_pred: predicted tensor
    :return: tensor with a single value
    """

    numerator = ops.sum(ops.abs(y_true - y_pred))
    denominator = ops.sum(ops.abs(y_true))

    return numerator / (denominator + K.epsilon())

sum_difference_metric.__name__ = "wMAPE"


class WeightedMeanAbsolutePercentageError(keras.metrics.Metric):
    def __init__(self, name="wmape", **kwargs):
        super().__init__(name=name, **kwargs)
        self.numerator = self.add_weight(name="numerator", initializer="zeros")
        self.denominator = self.add_weight(name="denominator", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        error = ops.sum(ops.abs(y_true - y_pred))
        scale = ops.sum(ops.abs(y_true))

        self.numerator.assign_add(error)
        self.denominator.assign_add(scale)

    def result(self):
        return self.numerator / (self.denominator + K.epsilon())

    def reset_state(self):
        self.numerator.assign(0.0)
        self.denominator.assign(0.0)
