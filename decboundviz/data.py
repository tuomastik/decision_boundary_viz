# -*- coding: utf-8 -*-

import numpy as np

from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification


X_TRAIN, X_TEST, X_TRAIN_TEST, Y_TRAIN, Y_TEST = None, None, None, None, None


def get_padded_range(array, padding=0.1):
    min_val = np.min(array) - padding * np.abs(np.min(array))
    max_val = np.max(array) + padding * np.abs(np.max(array))
    return min_val, max_val


def create_artificial(seed=85):
    x, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                               random_state=seed, n_clusters_per_class=1)
    x = StandardScaler().fit_transform(x)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=.4, random_state=seed)
    return x_train, x_test, y_train, y_test


def initialize(x_train, x_test, y_train, y_test):
    global X_TRAIN, X_TEST, X_TRAIN_TEST, Y_TRAIN, Y_TEST
    X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = x_train, x_test, y_train, y_test
    X_TRAIN_TEST = np.concatenate((x_train, x_test))


def get():
    return X_TRAIN, X_TEST, Y_TRAIN, Y_TEST
