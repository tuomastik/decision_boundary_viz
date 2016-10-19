# -*- coding: utf-8 -*-

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

from . import utils


MESH_STEP_SIZE = .05


def knn(n_neighbors=4, weights='uniform'):
    # n_neighbors = 4
    # weights = 'uniform'

    if weights == '0':
        weights = 'uniform'
    elif weights == '1':
        weights = 'distance'

    n_neighbors = int(n_neighbors)

    x_train, x_test, y_train, y_test = utils.get_data()

    clf = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights,
                               n_jobs=-1)
    clf.fit(x_train, y_train)

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    x_min, x_max = utils.get_padded_range(utils.X_TRAIN_TEST[:, 0])
    y_min, y_max = utils.get_padded_range(utils.X_TRAIN_TEST[:, 1])

    xx, yy = np.meshgrid(np.arange(x_min, x_max, MESH_STEP_SIZE),
                         np.arange(y_min, y_max, MESH_STEP_SIZE))
    z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    z = z.reshape(xx.shape)

    return z.tolist(), x_min, y_min, x_max-x_min, y_max-y_min
