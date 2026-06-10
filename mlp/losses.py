import numpy as np


def cross_entropy(y_true, y_pred):
    epsilon = 1e-12
    clipped_pred = np.clip(y_pred, epsilon, 1.0 - epsilon)
    return -np.mean(np.sum(y_true * np.log(clipped_pred), axis=1))

