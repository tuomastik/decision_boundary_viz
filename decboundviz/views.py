# -*- coding: utf-8 -*-

import json

from flask import render_template, request

from . import app, bokehscatter, data
from .classifiers import Classifiers


@app.route('/')
def index():
    data.initialize(*data.create_artificial())
    script, div, js_resources, css_resources = bokehscatter.create(
        default_clf=Classifiers.svm)
    return render_template(
        template_name_or_list='index.html', script=script, div=div,
        js_resources=js_resources, css_resources=css_resources)


@app.route("/get_classifier_info", methods=['POST'])
def get_classifier_info():
    # app.logger.debug("Sent by browser: %s", json.dumps(request.form))
    clf = Classifiers.get_clf_by_name(request.form['newly_selected_clf'])
    # app.logger.debug("Returning to client: %s", clf_info)
    return json.dumps(clf.get_info())


@app.route("/get_new_decision_boundary", methods=['POST'])
def get_new_decision_boundary():
    clf = Classifiers.get_clf_by_name(name=request.form['clf_name'])
    params = {request.form['param1_name']: request.form['param1_val'],
              request.form['param2_name']: request.form['param2_val']}
    clf.train(params=params)
    return json.dumps({
        'new_dec_bound': clf.get_decision_boundary()[0],
        'train_acc': '%.2f' % clf.get_train_accuracy(),
        'test_acc': '%.2f' % clf.get_test_accuracy()})
