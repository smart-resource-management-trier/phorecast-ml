import numpy
import tensorflow

def create_tf_dataset(data: tuple | numpy.ndarray, batch_size: int, shuffle: bool = False) -> tensorflow.data.Dataset:
    """
    Create a TensorFlow Dataset from a numpy array with specified batch size,
    including shuffling, prefetching, and caching for optimized performance.

    :param data: (np.ndarray or tuple): Single numpy array of features or tuple of numpy arrays
        (features, optional labels).
    :param batch_size: batch size for the dataset
    :param shuffle:
    :return: TensorFlow Dataset
    """

    if isinstance(data, tuple):
        features, labels = data
        # Check if labels are provided
        buffer_size = len(features)
        if labels.size > 0:
            dataset = tensorflow.data.Dataset.from_tensor_slices((features, labels))
        else:
            dataset = tensorflow.data.Dataset.from_tensor_slices(features)
    else:
        dataset = tensorflow.data.Dataset.from_tensor_slices(data)
        buffer_size = len(data)

    # Shuffle the dataset with a buffer size equal to the number of elements
    if shuffle:
        dataset = dataset.shuffle(buffer_size=buffer_size, reshuffle_each_iteration=True)

    # Batch the dataset with the specified batch size
    dataset = dataset.batch(batch_size)

    # Apply prefetching for performance optimization
    dataset = dataset.prefetch(tensorflow.data.experimental.AUTOTUNE)

    dataset = dataset.cache()
    return dataset