# -*- coding: utf-8 -*-

import json

from flask import render_template, request

from . import app, bokehscatter, decisionboundary


@app.route('/')
def index():

    script, div, js_resources, css_resources = bokehscatter.create()

    return render_template(
        template_name_or_list='index.html',
        script=script,
        div=div,
        js_resources=js_resources,
        css_resources=css_resources)


@app.route("/get_classifier_info", methods=['POST'])
def get_classifier_info():
    clf_info = bokehscatter.CLASSIFIERS[request.form['newly_selected_clf']]
    return json.dumps(clf_info)


@app.route("/get_new_decision_boundary", methods=['POST'])
def get_new_decision_boundary():
    app.logger.info("Browser sent the following: %s", json.dumps(request.form))
    clf_name = request.form['clf_name']
    params = {
        request.form['attr1_name'].strip().replace(':', ''):
            request.form['attr1_val'],
        request.form['attr2_name'].strip().replace(':', ''):
            request.form['attr2_val']}
    # app.logger.debug("Returning to client: %s", clf_info)
    if clf_name == 'k-NN':
        return json.dumps(decisionboundary.knn(**params))
    else:
        return json.dumps([1, 2, 3])
