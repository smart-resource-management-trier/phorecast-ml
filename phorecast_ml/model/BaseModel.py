import numpy

class BaseModel(object):
    def __init__(self):
        self.__model = None

    def train(self, train_X: numpy.ndarray, train_y: numpy.ndarray, test_X: numpy.ndarray, test_y: numpy.ndarray):
        raise NotImplementedError()

    def predict(self, X: numpy.ndarray):
        raise NotImplementedError()