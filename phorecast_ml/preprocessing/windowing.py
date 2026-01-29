import numpy
import pandas

def windowing(data: pandas.DataFrame, window_size: int = 24, stride: int = 6, max_missing: int = 6) -> list[pandas.DataFrame]:
    """
    Creates windows from the given data (Has to be uniform data with ts index)

    :param data: dataframe with a datetime index and hourly data points
    :param window_size: size of the window in hours
    :param stride: stepover of the window in hours
    :param max_missing: maximum number of missing values in a window

    :return: list of windows (Dataframes)
    :raises ValueError: if window_size is 0, stride is 0, index has duplicates,
        max_missing > window_size, window size mismatch
    """
    if window_size <= 0:
        raise ValueError("window_size cannot be 0")
    if stride <= 0:
        raise ValueError("stride cannot be 0")

    if any(data.index.duplicated()):
        raise ValueError("cant window data with duplicated timestamps")

    data.sort_index(inplace=True)
    min_ts = data.index.min()
    max_ts = data.index.max()

    window_lower = min_ts
    window_upper = min_ts + pandas.Timedelta(hours=window_size)
    windows = []

    # fill empty timesteps with this value
    filler = 0

    while window_upper < max_ts + pandas.Timedelta(hours=window_size):
        window = data.loc[window_lower:window_upper - pandas.Timedelta(seconds=1)].copy()

        if len(window) == window_size:  # complete window
            windows.append(window.sort_index())

        # window with missing values but not too many
        elif window_size - len(window) <= max_missing:
            for i in pandas.date_range(window_lower, window_upper - pandas.Timedelta(seconds=1), freq="1h"):  # insert missing timestamps
                if i not in window.index:
                    window.loc[i] = filler
            window.sort_index(inplace=True)

            #window.infer_objects(copy=False).fillna(0, inplace=True)  # fill missing values with 0
            window = window.infer_objects().fillna(0)
            windows.append(window)

        window_lower += pandas.Timedelta(hours=stride)
        window_upper += pandas.Timedelta(hours=stride)

    for w in windows:
        if len(w) > window_size:
            raise ValueError("Window size mismatch")

    return windows


def get_dataset_from_windows(windows: list[pandas.DataFrame], target: str = "Target") \
         -> tuple[tuple[numpy.ndarray, numpy.ndarray], list]:
    """
    Takes a list of pandas dataframes, which are timeframes, and converts them to a dataset
    (np-array)

    :param windows: list of pandas dataframes
    :param target: Target column name
    :return: Tuple of arrays (X,y) with X being the parameters and y being the targets and a list
        of indices (datetime index). Arrays have shape (number of windows, window size, number of
        parameters)
    """

    x, y, index = [], [], []
    window_size = len(windows[0])
    columns = [col for col in windows[0].columns if col != target]

    for w in windows:
        x.append(w[columns].values.astype(numpy.float32))
        if target in w.columns:
            y.append(numpy.array(w[target].values.astype(numpy.float32)).reshape(window_size, 1))
        index.append(w.index.values)

    x = numpy.array(x)
    y = numpy.array(y)
    return (x, y), index