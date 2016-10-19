import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification


def get_data(seed=85):
    x, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                               random_state=seed, n_clusters_per_class=1)
    x = StandardScaler().fit_transform(x)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.4)
    return x_train, x_test, y_train, y_test


MESH_STEP_SIZE = .05
X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = get_data()


def get_padded_range(array, padding=0.1):
    min_val = np.min(array) - padding * np.abs(np.min(array))
    max_val = np.max(array) + padding * np.abs(np.max(array))
    return min_val, max_val


def knn(n_neighbors=4, weights='uniform'):
    # n_neighbors = 4
    # weights = 'uniform'

    if weights == '0':
        weights = 'uniform'
    elif weights == '1':
        weights = 'distance'

    n_neighbors = int(n_neighbors)

    # x_train, x_test, y_train, y_test = get_data()
    clf = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights,
                               n_jobs=-1)
    clf.fit(X_TRAIN, Y_TRAIN)

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    x_min, x_max = get_padded_range(X_TRAIN[:, 0])
    y_min, y_max = get_padded_range(X_TRAIN[:, 1])

    xx, yy = np.meshgrid(np.arange(x_min, x_max, MESH_STEP_SIZE),
                         np.arange(y_min, y_max, MESH_STEP_SIZE))
    z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    z = z.reshape(xx.shape)

    return z.tolist(), x_min, y_min, x_max-x_min, y_max-y_min
