# Machine Learning models optimized for PV forecast

`phorecast-ml` is a Python library for **photovoltaic (PV) power forecasting** built on top of **Keras 3**.  
It provides **predefined deep learning models**, a **scikit-learn–like API** for time-series forecasting.

This package serves as the **machine learning (ML) foundation of the Phorecast project**, providing reusable, well-tested ML components that can be used independently or as part of the full Phorecast forecasting pipeline.

## Features
- **scikit-learn–style API** (`fit`, `predict`)
- Predefined PV forecasting models (e.g. LSTM)
- Flexible configuration of predefined forecasting models
  - Optimizers
  - Loss functions
  - Metrics
  - HP (e.g. network depth, network width, ...)
- **PV-specific metrics** (e.g. **wMAPE**, ...)
- Integrated methods for calculating the position of the sun
- Integrated filtering method for invalid data (e.g., incorrect PV data, incorrect solar forecasts based on sun position)
- Integrated methods for dividing a data set into train-test data sets

## Installation
The latest versions of `phorecast-ml` are published on pypi.org, allowing for easy installation via pip using the following command.
```bash
pip install phorecast-ml
```

## Usage
The following example shows how a simple LSTM network can be created and trained using the package. The hyperparameters can be selected as desired.

```python
import phorecast_ml

model = phorecast_ml.model.LSTM(
    units=128,
    depth=4,
    learning_rate=1e-3,
    epochs=100,
    patience=10,
    metrics=["mae", "wmape"],
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```
