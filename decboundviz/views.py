# -*- coding: utf-8 -*-

import json

from flask import render_template, request

from . import app, bokehscatter, data, classifiers


@app.route('/')
def index():
    data.initialize(*data.create_artificial())
    script, div, js_resources, css_resources = bokehscatter.create(
        *data.get())
    return render_template(
        template_name_or_list='index.html', script=script, div=div,
        js_resources=js_resources, css_resources=css_resources)


@app.route("/get_classifier_info", methods=['POST'])
def get_classifier_info():
    clf_info = classifiers.CLFS[request.form['newly_selected_clf']]
    return json.dumps(clf_info)


@app.route("/get_new_decision_boundary", methods=['POST'])
def get_new_decision_boundary():
    app.logger.info("Browser sent the following: %s", json.dumps(request.form))
    clf_name = request.form['clf_name']
    params = {
        # Bokeh automatically adds ':' and whitespace to Slider label
        request.form['param1_name'].strip().replace(':', ''):
            request.form['param1_val'],
        request.form['param2_name'].strip().replace(':', ''):
            request.form['param2_val']}
    # app.logger.debug("Returning to client: %s", clf_info)
    if clf_name == 'k-NN':
        clf = classifiers.train_knn(**params)
    elif clf_name == 'SVM (RBF kernel)':
        clf = classifiers.train_svm(**params)
    elif clf_name == 'Random forest':
        clf = classifiers.train_rf(**params)
    else:
        return json.dumps([1, 2, 3])
    return json.dumps({
        'new_dec_bound': classifiers.get_decision_boundary(clf=clf)[0],
        'accuracy': '%.2f' % classifiers.get_accuracy(clf=clf)})
