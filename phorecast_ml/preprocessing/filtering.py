import numpy
import pandas
import sklearn

def solar_position_filter(data: pandas.DataFrame, independent_variables: list[str], target: str):
    """
    This function filters the data in 3 steps
    :param data: dataframe with the data
    :param independent_variables:
    :param target:

    :return: dataframe with the filtered data
    """

    if len(data.columns.intersection(independent_variables)) < len(independent_variables):
        return data

    if target not in data.columns:
        return data

    if "elevation" in data.columns:
        # Rule 1: If elevation < -10, target has to be 0
        data = data[~((data["elevation"] < -10) & (data[target] > 0))].copy()

        # Rule 2: Remove rows where elevation > 10 and target is 0
        data = data[~((data["elevation"] > 10) & (data[target] == 0))].copy()

    # Rule 3: Remove outliers with Linear Regression
    x = data[independent_variables]
    y = data[target]

    # Fit the model and predict
    model = sklearn.linear_model.LinearRegression()
    model.fit(x, y)
    y_pred = model.predict(x)

    # calculate residuals
    data["__residuals"] = y - y_pred
    residual_std = numpy.std(data["__residuals"])
    residual_mean = numpy.mean(data["__residuals"])
    threshold = 5 * residual_std

    condition = (
            (data["__residuals"] <= (residual_mean + threshold))
            & (data["__residuals"] >= (residual_mean - threshold))
    )

    data_filtered = data.loc[condition].copy()

    data_filtered = data_filtered.drop(columns=["__residuals"])

    return data_filtered