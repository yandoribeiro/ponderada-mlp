import numpy as np


def one_hot(labels, num_classes):
    labels = np.asarray(labels, dtype=int).reshape(-1)

    if np.any(labels < 0) or np.any(labels >= num_classes):
        raise ValueError("Rotulos fora do intervalo de classes.")

    encoded = np.zeros((labels.shape[0], num_classes))
    encoded[np.arange(labels.shape[0]), labels] = 1
    return encoded
