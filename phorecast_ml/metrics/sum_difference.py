"""
This file contains custom keras objects.
"""

import keras
import tensorflow as tf


@keras.saving.register_keras_serializable(package="Metrics", name="sum_difference")
def sum_difference_metric(y_true, y_pred):
    """
    This metric is used to calculate the mae of the sum in the second dimension (timesteps) of the
    output tensor.
    :param y_true: ground truth tensor
    :param y_pred: predicted tensor
    :return: tensor with a single value
    """
    return tf.math.reduce_mean(
        tf.math.abs(tf.math.reduce_sum(y_true, axis=1) - tf.math.reduce_sum(y_pred, axis=1)))


sum_difference_metric.__name__ = 'sum_difference'
