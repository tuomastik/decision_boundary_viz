# -*- coding: utf-8 -*-

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from bokeh.models import Slider


from . import data


class Classifier:

    MESH_STEP_SIZE = .05  # In the decision boundary

    def __init__(self, name, clf,
                 slider1, slider1_val_converter,
                 slider2, slider2_val_converter):
        self.name = name
        self.clf = clf
        self.slider1 = slider1
        self.slider1_val_converter = slider1_val_converter
        self.slider2 = slider2
        self.slider2_val_converter = slider2_val_converter

    def get_info(self):
        info = {}
        for i, slider in enumerate([self.slider1, self.slider2], start=1):
            slider_nr = 'slider%s' % i
            info[slider_nr] = {}
            for param_name in ['title', 'start', 'end', 'step', 'value']:
                info[slider_nr][param_name] = getattr(slider, param_name)
        return info

    def formalize_params(self, params):
        # Transform displayed parameter names (dict keys) and
        # parameter values (dict values) to the ones
        # that corresponding classifier instance accepts
        for param_name, param_value in params.items():
            for slider, slider_converter in zip(
                    [self.slider1, self.slider2],
                    [self.slider1_val_converter, self.slider2_val_converter]):
                if param_name == slider.title:
                    # Convert parameter name:
                    # slider.title stores the visible parameter name and
                    # slider.name stores the actual parameter name
                    old_key, new_key = param_name, slider.name
                    params[new_key] = params.pop(old_key)
                    # Convert parameter value
                    params[new_key] = slider_converter(params[new_key])
        return params

    def train(self, params):
        # Formalize parameters
        params = self.formalize_params(params=params)
        # Update parameter values
        for param_name, param_val in params.items():
            setattr(self.clf, param_name, param_val)
        self.clf.fit(data.X_TRAIN, data.Y_TRAIN)

    def get_decision_boundary(self):
        # To get the decision boundary, predict class for each point
        # in the mesh [x_min, x_max]x[y_min, y_max]
        x_min, x_max = data.get_padded_range(data.X_TRAIN_TEST[:, 0])
        y_min, y_max = data.get_padded_range(data.X_TRAIN_TEST[:, 1])
        xx, yy = np.meshgrid(np.arange(x_min, x_max, self.MESH_STEP_SIZE),
                             np.arange(y_min, y_max, self.MESH_STEP_SIZE))
        z = self.clf.predict(np.c_[xx.ravel(), yy.ravel()])
        z = z.reshape(xx.shape)
        return z.tolist(), x_min, y_min, x_max - x_min, y_max - y_min

    def get_train_accuracy(self):
        return accuracy_score(y_true=data.Y_TRAIN,
                              y_pred=self.clf.predict(data.X_TRAIN)) * 100.0

    def get_test_accuracy(self):
        return accuracy_score(y_true=data.Y_TEST,
                              y_pred=self.clf.predict(data.X_TEST)) * 100.0


class Converters:
    def __init__(self):
        pass

    @staticmethod
    def to_float(value):
        return np.float(value)

    @staticmethod
    def to_int(value):
        return np.int(value)

    @staticmethod
    def knn_weight(value):
        if value == '0':
            return 'uniform'
        elif value == '1':
            return'distance'
        else:
            raise(Exception("Unknown k-NN weight value: %s" % value))


class Classifiers:
    knn = Classifier(name='k-Nearest Neighbors',
                     clf=KNeighborsClassifier(),
                     slider1=Slider(title='Number of neighbors',
                                    name='n_neighbors',
                                    start=1, end=50, step=1, value=4),
                     slider1_val_converter=Converters.to_int,
                     slider2=Slider(title='Weight function',
                                    name='weights',
                                    start=0, end=1, step=1, value=1),
                     slider2_val_converter=Converters.knn_weight)
    rf = Classifier(name='Random forest',
                    clf=RandomForestClassifier(),
                    slider1=Slider(title='The number of trees in the forest',
                                   name='n_estimators',
                                   start=1, end=501, step=20, value=11),
                    slider1_val_converter=Converters.to_int,
                    slider2=Slider(title='The maximum depth of trees',
                                   name='max_depth',
                                   start=1, end=21, step=1, value=1),
                    slider2_val_converter=Converters.to_int)
    svm = Classifier(name='Support vector machine (RBF kernel)',
                     clf=SVC(kernel='rbf'),
                     slider1=Slider(title='Cost (C)',
                                    name='C',
                                    start=0.1, end=10 ** 4, step=0.1,
                                    value=10),
                     slider1_val_converter=Converters.to_float,
                     slider2=Slider(title='Gamma', name='gamma',
                                    start=0.1, end=400, step=0.1, value=0.1),
                     slider2_val_converter=Converters.to_float)

    def __init__(self):
        pass

    @staticmethod
    def get_clf_by_name(name):
        for _, var in vars(Classifiers).items():
            if isinstance(var, Classifier) and var.name == name:
                return var
        return None

    @staticmethod
    def get_classifier_names():
        classifier_names = []
        for _, var in vars(Classifiers).items():
            if isinstance(var, Classifier):
                classifier_names.append(var.name)
        return classifier_names
