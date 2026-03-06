# Phorecast ML

[![Python](https://img.shields.io/badge/python-3.12+-blue.svg)]()
[![License](https://img.shields.io/badge/license-see%20LICENSE-green.svg)]()
[![Deep Learning](https://img.shields.io/badge/ML-TensorFlow%20%7C%20Keras-orange.svg)]()
[![PV Forecasting](https://img.shields.io/badge/domain-photovoltaics-yellow.svg)]()

A Python library for **photovoltaic (PV) power forecasting** using **deep learning models** and specialized **time-series preprocessing tools**.

The library provides modular building blocks for:

- preprocessing PV time-series data
- generating training datasets
- training deep learning forecasting models
- generating PV power forecasts

The current implementation includes an **LSTM-based forecasting model** built on **TensorFlow / Keras**.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture Overview](#architecture-overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Example Workflow](#example-workflow)
- [Dependencies](#dependencies)
- [Current Scope](#current-scope)
- [Running Tests](#running-tests)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

`phorecast_ml` is designed as a **machine learning foundation layer for photovoltaic forecasting**.

It focuses on:

- preparing PV time-series datasets
- creating structured training data
- training deep learning forecasting models

The design follows a **simple API pattern similar to scikit-learn**, with methods such as `fit()` and `predict()`.

This makes the models easy to integrate into forecasting pipelines.

---

## Features

### Forecasting Models

- LSTM-based neural network model
- configurable architecture
- training via TensorFlow/Keras

### Data Processing

Tools for preparing time-series data:

- window generation
- dataset creation
- train/test splitting
- TensorFlow dataset generation

### Solar Data Utilities

- solar position calculation using **pvlib**
- solar-based data filtering
- preprocessing utilities for photovoltaic datasets

### Custom Evaluation

Includes custom metrics and loss functions:

- `wmape`
- `sum_difference`

---

## Architecture Overview

The library is structured into **four main components**:

```
Data Input
   │
   ▼
Preprocessing
(windowing, filtering, solar features)
   │
   ▼
Dataset Generation
(X / y creation)
   │
   ▼
Deep Learning Model
(LSTM)
   │
   ▼
Predictions
```

The architecture separates **data preparation**, **model training**, and **forecast generation**.

---

## Project Structure

```text
phorecast_ml/
  __init__.py

  model/
    BaseModel.py
    LSTM.py
    __init__.py

  preprocessing/
    dataset.py
    dataset_splitting.py
    filtering.py
    solar.py
    windowing.py

  metrics/
    sum_difference.py
    weighted_mean_absolute_percentage_error.py

  losses/
    sum_difference_loss.py

tests/
  static.py
  test_dataset.py
  test_dataset.csv

pyproject.toml
requirements.txt
README.md
LICENSE
```

---

## Installation

### Requirements

- Python >= 3.12

### Install from repository

```bash
git clone https://github.com/<organization>/phorecast-ml.git
cd phorecast-ml
pip install .
```

### Install from PyPI (if published)

```bash
pip install phorecast-ml
```

---

## Usage

### Basic Model Training

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

predictions = model.predict(X_test)
```

---

## Example Workflow

Typical forecasting pipeline:

1. Prepare PV time-series data
2. Attach solar position features
3. Filter invalid or noisy data
4. Generate time windows
5. Convert windows into training datasets
6. Train the forecasting model
7. Generate predictions

### Create Time Windows

```python
from phorecast_ml.preprocessing.windowing import windowing, get_dataset_from_windows

windows = windowing(dataframe, window_size=24, stride=6, max_missing=6)

(X, y), indices = get_dataset_from_windows(
    windows,
    target="Target"
)
```

### Attach Solar Features

```python
from phorecast_ml.preprocessing.solar import attach_solar_positions

dataframe = attach_solar_positions(
    data=dataframe,
    latitude=0.0,
    longitude=0.0,
    height=0.0
)
```

### Filter Dataset

```python
from phorecast_ml.preprocessing.filtering import solar_position_filter

filtered = solar_position_filter(
    data=dataframe,
    independent_variables=["feature_1", "feature_2", "elevation"],
    target="Target"
)
```

### Train/Test Split

```python
from phorecast_ml.preprocessing.dataset_splitting import split_windows

train_windows, test_windows = split_windows(
    windows,
    test_ratio=0.25,
    weeks_in_test=1,
    factor=7,
    distinct=False
)
```

---

## Dependencies

Primary dependencies:

- numpy
- pandas
- tensorflow
- keras
- pvlib

Additional libraries used in the code:

- scikit-learn (`sklearn.linear_model.LinearRegression`)

---

## Current Scope

The current implementation focuses on an **LSTM-based forecasting model** for photovoltaic power prediction.

The repository currently provides tools for:

- preprocessing photovoltaic time-series data
- generating training datasets
- training deep learning forecasting models
- generating predictions

Future improvements may include:

- additional forecasting model architectures
- expanded preprocessing utilities
- improved documentation
- additional unit tests

---

## Running Tests

Tests can be executed using:

```bash
pytest
```

---

## Contributing

Contributions are welcome.

Typical improvements include:

- improving documentation
- adding new forecasting models
- improving preprocessing pipelines
- adding unit tests

Suggested workflow:

```
fork repository
create feature branch
commit changes
submit pull request
```

---

## License

This project is licensed under the terms specified in the `LICENSE` file.