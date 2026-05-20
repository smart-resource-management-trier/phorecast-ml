import numpy
import tensorflow
import keras

import phorecast_ml.metrics.sum_difference
import phorecast_ml.metrics.weighted_mean_absolute_percentage_error
import phorecast_ml.model.BaseModel
import phorecast_ml.preprocessing.dataset

class LSTM(phorecast_ml.model.BaseModel):
    def __init__(
            self,
            units: int = 128,
            depth: int = 5,
            dropout: float = 0.2,
            optimizer: keras.optimizers.Optimizer | str = "adam",
            optimizer_params: dict | None = None,
            learning_rate: float = 1e-3,
            loss: str | keras.losses.Loss  = "mean_absolute_error",
            metrics: list | None = None,
            batch_size: int = 32,
            epochs: int = 100,
            patience: int = 30,
            random_state: int | None = None,
    ):
        super().__init__()

        self.units = units
        self.depth = depth
        self.dropout = dropout
        self.optimizer = optimizer
        self.optimizer_params = optimizer_params or {}
        self.learning_rate = learning_rate
        self.loss = loss
        self.metrics = metrics
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.random_state = random_state

    def fit(self, X: numpy.ndarray, y: numpy.ndarray, X_val: numpy.ndarray = None, y_val: numpy.ndarray = None):
        return self.train(X, y, X_val, y_val)


    def train(self, X: numpy.ndarray, y: numpy.ndarray, X_val: numpy.ndarray = None, y_val: numpy.ndarray = None):
        train_data = phorecast_ml.preprocessing.dataset.create_tf_dataset((X, y), 32, shuffle=True)

        test_data = None
        if X_val is not None and y_val is not None:
            test_data = phorecast_ml.preprocessing.dataset.create_tf_dataset((X_val, y_val), 32)

        self.__model = self._build_model(train_data)

        early_stopping_callback = keras.callbacks.EarlyStopping(patience=self.patience)

        history = self.__model.fit(train_data, epochs=self.epochs, validation_data=test_data,
                            callbacks=[early_stopping_callback], verbose=1)

        return history


    def predict(self, X: numpy.ndarray):
        if not hasattr(self, "_LSTM__model"):
            raise RuntimeError()

        predict_data = phorecast_ml.preprocessing.dataset.create_tf_dataset(
            X,
            self.batch_size,
            shuffle=False
        )

        return self.__model.predict(predict_data, verbose=0)


    def save(self, filepath, overwrite=True, **kwargs):
        if not hasattr(self, "_LSTM__model"):
            raise RuntimeError()

        self.__model.save(filepath=filepath, overwrite=overwrite, **kwargs)


    def _build_optimizer(self) -> keras.optimizers.Optimizer:
        lr = self.learning_rate
        params = dict(self.optimizer_params)

        if issubclass(type(self.optimizer), keras.optimizers.Optimizer):
            return self.optimizer

        match self.optimizer.lower():
            case "adam":
                return keras.optimizers.Adam(learning_rate=lr, **params)

            case "sgd":
                return keras.optimizers.SGD(learning_rate=lr, **params)

            case "rmsprop":
                return keras.optimizers.RMSprop(learning_rate=lr, **params)

            case _:
                raise ValueError(f"Unknown optimizer '{self.optimizer}'")


    def _build_loss(self):
        """
        Resolve loss configuration to a keras-compatible loss.
        """
        if isinstance(self.loss, str) or isinstance(self.loss, keras.losses.Loss) or callable(self.loss):
            return self.loss

        raise TypeError(
            "loss must be str, keras.losses.Loss, or callable. "
            f"Got {type(self.loss)}."
        )


    def _build_metrics(self) -> list:
        """
        Resolve metrics configuration to keras-compatible metrics.
        """
        # Default
        if self.metrics is None:
            return ["mean_absolute_error"]

        resolved_metrics = []

        for metric in self.metrics:
            if isinstance(metric, str) or isinstance(metric, keras.metrics.Metric) or callable(metric):
                resolved_metrics.append(metric)

            else:
                raise TypeError(
                    "Each metric must be str, keras.metrics.Metric, or callable. "
                    f"Got {type(metric)}."
                )

        return resolved_metrics


    def _build_model(self, data: tensorflow.data.Dataset) -> keras.models.Sequential:
        _, window_size, num_of_features = data.element_spec[0].shape

        model = keras.models.Sequential()
        batch_size = self.batch_size if data is not None else 1

        # create input (InputLayer, MaskingLayer, NormalizationLayer)
        model.add(keras.layers.Input((None, num_of_features), batch_size=batch_size))

        # create and adapt normalization layer
        normalization_layer = keras.layers.Normalization(axis=-1)
        normalization_layer.adapt(data=data.map(lambda x, y: x))
        model.add(normalization_layer)


        # create body of model (LSTM layers)
        for _ in range(self.depth):
            model.add(keras.layers.LSTM(units=self.units,
                                        return_sequences=True,
                                        dropout=self.dropout)
            )

        # create output layer (DenseLayer, inverted NormalizationLayer)
        model.add(keras.layers.Dense(units=1, activation="linear"))

        # create and adapt normalization layer
        normalization_layer_output = keras.layers.Normalization(axis=-1, invert=True)
        normalization_layer_output.adapt(data=data.map(lambda x, y: y))
        model.add(normalization_layer_output)

        # compile model
        model.compile(optimizer=self._build_optimizer(),
                      loss=self._build_loss(),
                      metrics=self._build_metrics(),
        )

        return model
