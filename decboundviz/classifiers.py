# -*- coding: utf-8 -*-

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


from . import data


MESH_STEP_SIZE = .05  # In the decision boundary
CLFS = {
    'Random forest': {
        'slider1': {'title': 'n_estimators', 'start': 1, 'end': 501,
                    'step': 20, 'value': 11},
        'slider2': {'title': 'max_depth', 'start': 1, 'end': 51,
                    'step': 1, 'value': 1},
    },
    'k-NN': {
        'slider1': {'title': 'n_neighbors', 'start': 1, 'end': 50,
                    'step': 1, 'value': 4},
        'slider2': {'title': 'weights', 'start': 0, 'end': 1,
                    'step': 1, 'value': 1},
    },
    'SVM (RBF kernel)': {
        'slider1': {'title': 'c', 'start': 0.1, 'end': 10**4,
                    'step': 0.1, 'value': 10},
        'slider2': {'title': 'gamma', 'start': 0.1, 'end': 400,
                    'step': 0.1, 'value': 0.1},
    }
}
DEFAULT_CLF = 'k-NN'


def train_knn(n_neighbors=4, weights='uniform'):
    if str(weights) == '0':
        weights = 'uniform'
    elif str(weights) == '1':
        weights = 'distance'
    x_train, x_test, y_train, y_test = data.get()
    return KNeighborsClassifier(n_neighbors=int(n_neighbors), weights=weights,
                                n_jobs=1).fit(x_train, y_train)


def train_svm(c=0.1, gamma=0.1):
    x_train, x_test, y_train, y_test = data.get()
    return SVC(kernel='rbf', C=float(c), gamma=float(gamma)).fit(
        x_train, y_train)


def train_rf(n_estimators, max_depth):
    x_train, x_test, y_train, y_test = data.get()
    return RandomForestClassifier(
        n_estimators=int(n_estimators), max_depth=int(max_depth),
        n_jobs=1).fit(x_train, y_train)


def get_accuracy(clf):
    x_train, x_test, y_train, y_test = data.get()
    return accuracy_score(y_true=y_test, y_pred=clf.predict(x_test)) * 100.0


def get_decision_boundary(clf):
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    x_min, x_max = data.get_padded_range(data.X_TRAIN_TEST[:, 0])
    y_min, y_max = data.get_padded_range(data.X_TRAIN_TEST[:, 1])
    xx, yy = np.meshgrid(np.arange(x_min, x_max, MESH_STEP_SIZE),
                         np.arange(y_min, y_max, MESH_STEP_SIZE))
    z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    z = z.reshape(xx.shape)
    return z.tolist(), x_min, y_min, x_max-x_min, y_max-y_min
