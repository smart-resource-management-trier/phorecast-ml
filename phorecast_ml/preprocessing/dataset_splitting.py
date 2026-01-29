import random

import pandas

def split_windows(windows, test_ratio=0.25, weeks_in_test=1, factor=7, distinct: bool = False) \
        -> tuple[list[pandas.DataFrame], list[pandas.DataFrame]]:
    """
    Splits a list of windows (dataframes) into a train and test set.

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

    train, test = [], []
    test_selection = [False] * len(windows)
    max_index = max(window.index.max() for window in windows)

    for index, window in enumerate(windows):
        if window.index.max() > max_index - pandas.Timedelta(weeks=weeks_in_test):
            test_selection[index] = True

    while test_selection.count(True) / len(test_selection) < test_ratio:
        random_index = random.randint(0, len(test_selection) - factor)
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
        for test_window in train:
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