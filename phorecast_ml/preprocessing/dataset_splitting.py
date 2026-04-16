import numpy.random
import pandas

import phorecast_ml

def split_windows(windows,
                  strategy="random",
                  distinct: bool = False,
                  rng: numpy.random.Generator | int = None,
                  **kwargs
):
    if rng is None:
        rng =  phorecast_ml.get_rng()
    elif isinstance(rng, int):
        rng = numpy.random.default_rng(rng)

    match strategy:
        case "time":
            train, test = _time_split(windows, **kwargs)

        case "random":
            train, test = _random_split(windows, rng=rng, **kwargs)

        case "seasonal":
            train, test = _seasonal_split(windows, rng=rng, **kwargs)

        case _:
            raise ValueError(f"Unknown strategy: {strategy}")

    if distinct:
        train = remove_overlap(train, test)

    return train, test


def _time_split(windows, weeks_in_test=1, **_):
    import pandas as pd

    windows = sorted(windows, key=lambda w: w.index.min())

    max_time = max(w.index.max() for w in windows)
    cutoff = max_time - pd.Timedelta(weeks=weeks_in_test)

    train, test = [], []

    for w in windows:
        if w.index.max() < cutoff:
            train.append(w)
        elif w.index.min() >= cutoff:
            test.append(w)

    return train, test

def _random_split(windows, rng, test_ratio=0.25, factor=7, **_):
    n = len(windows)
    n_test = int(n * test_ratio)

    indices = list(range(n))

    if factor <= 1:
        rng.shuffle(indices)
        test_index = set(indices[:n_test])
    else:
        # block sampling
        block_starts = list(range(0, n - factor + 1))
        rng.shuffle(block_starts)

        test_index = set()

        for start in block_starts:
            for i in range(factor):
                test_index.add(start + i)

            if len(test_index) >= n_test:
                break

    return _get_datasets_from_index(windows, test_index)

def _seasonal_split(windows, rng, test_ratio=0.25, **_):
    from collections import defaultdict

    def get_season(month):
        return (month % 12) // 3

    groups = defaultdict(list)

    for i, w in enumerate(windows):
        month = w.index.min().month
        season = get_season(month)
        groups[season].append(i)

    test_index = set()

    for indices in groups.values():
        rng.shuffle(indices)
        n = int(len(indices) * test_ratio)
        test_index.update(indices[:n])

    return _get_datasets_from_index(windows, test_index)

def _get_datasets_from_index(windows, test_indices):
    test_indices = set(test_indices)

    train, test = [], []

    for i, w in enumerate(windows):
        if i in test_indices:
            test.append(w)
        else:
            train.append(w)

    return train, test

def remove_overlap(train, test):
    def overlaps(a_min, a_max, b_min, b_max):
        return a_min <= b_max and a_max >= b_min

    test_intervals = [
        (window.index.min(), window.index.max())
        for window in test
    ]

    filtered_train = []

    for window in train:
        train_window_min = window.index.min()
        train_window_max = window.index.max()

        if not any(overlaps(train_window_min, train_window_max, test_window_min, test_window_max) for test_window_min, test_window_max in test_intervals):
            filtered_train.append(window)

    return filtered_train



def split_windows_old(windows, test_ratio=0.25, weeks_in_test=1, factor=7, distinct: bool = False) \
        -> tuple[list[pandas.DataFrame], list[pandas.DataFrame]]:
    """
    Splits a list of windows (dataframes) into a train and test set.

    :param rng:
    :param seed:
    :param distinct: if true, train windows that overlap with test windows will be removed
    :param windows: windows to split
    :param test_ratio: ratio of test windows if test_ratio is 0.25 and there are 4 windows,
        1 will be test and 3 will be train.
    :param weeks_in_test: sets how many weeks, counting from the end of the dataset, will
        automatically be in the test set.
    :param factor: factor decides how many windows will be extracted together to minimize the
        intersection of train and test.
    :return: two lists of windows (train, test)
    """


    rng = numpy.random.default_rng(42)

    train, test = [], []
    test_selection = [False] * len(windows)
    max_index = max(window.index.max() for window in windows)

    for index, window in enumerate(windows):
        if window.index.max() > max_index - pandas.Timedelta(weeks=weeks_in_test):
            test_selection[index] = True

    while test_selection.count(True) / len(test_selection) < test_ratio:
        random_index = rng.integers(0, max(1, len(test_selection) - factor + 1))
        for i in range(factor):
            test_selection[random_index + i] = True

    for index, test_flag in enumerate(test_selection):
        if test_flag:
            test.append(windows[index])
        else:
            train.append(windows[index])



    # if distinct is true, remove all train windows that overlap with test windows
    train_elimination = [False] * len(train)
    if distinct:
        for test_window in test:
            test_min_index = test_window.index.min()
            test_max_index = test_window.index.max()
            for index, train_window in enumerate(train):
                train_min_index = train_window.index.min()
                train_max_index = train_window.index.max()
                if ((train_min_index <= test_min_index <= test_max_index)
                        or (train_min_index <= train_max_index <= test_max_index)):
                    train_elimination[index] = True

    train = [train_window for index, train_window in enumerate(train) if
             not train_elimination[index]]

    return train, test