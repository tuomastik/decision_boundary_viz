# -*- coding: utf-8 -*-

import numpy as np
from bokeh.plotting import figure
from bokeh.models import (ColumnDataSource, CustomJS, Select, Slider,
                          HoverTool, CheckboxGroup, Div)
from bokeh.embed import components
from bokeh.resources import INLINE
from bokeh.layouts import column, row

from . import classifiers, data


def plot_decision_region(fig):
    clf = classifiers.train_knn(
        n_neighbors=classifiers.CLFS['k-NN']['slider1']['value'],
        weights=classifiers.CLFS['k-NN']['slider2']['value'])
    d, x_min, y_min, dw, dh = classifiers.get_decision_boundary(clf=clf)
    source_region = ColumnDataSource(data={'d': [d]})
    region = fig.image(image='d', x=x_min, y=y_min,
                       dw=dw, dh=dh, source=source_region,
                       palette=['#DC9C76', '#D6655A'])
    return region, source_region


def create_sources(data_points, classes):
    """ Create one ColumnDataSource for examples in each class. """
    sources = []
    for unique_class in np.sort(np.unique(classes)):
        class_example_ix = np.where(classes == unique_class)[0]
        sources.append(
            ColumnDataSource(data={'x': data_points[class_example_ix, 0],
                                   'y': data_points[class_example_ix, 1]}))
    return sources


def create(x_train, x_test, y_train, y_test):

    # Create ColumnDataSources to store the examples in
    example_sources_train = create_sources(x_train, y_train)
    example_sources_test = create_sources(x_test, y_test)

    # Figure
    # -------------------------------------------------------------------------
    tools = [
        HoverTool(tooltips=[("Index", "$index"), ("X", "@x"), ("Y", "@y")]),
        "pan, wheel_zoom, box_select, lasso_select, reset, save"]
    fig = figure(tools=tools, plot_width=1110, plot_height=600,
                 x_range=data.get_padded_range(data.X_TRAIN_TEST[:, 0]),
                 y_range=data.get_padded_range(data.X_TRAIN_TEST[:, 1]),
                 toolbar_location="right", background_fill_color="#fafafa",
                 active_drag="pan", active_scroll="wheel_zoom", logo=None)
    fig.xaxis.visible = False
    fig.yaxis.visible = False
    fig.border_fill_color = None

    # Decision region
    # -------------------------------------------------------------------------
    region, source_region = plot_decision_region(fig)

    # Scatter plot
    # -------------------------------------------------------------------------
    colors = ["#D9CFB0", "#42282E"]
    markers = ["circle", "triangle"]
    scatters = {}
    for i, (train_source, test_source, color, marker) in enumerate(zip(
            example_sources_train, example_sources_test, colors, markers)):
        scatters['train_scatter_class_%s' % i] = fig.scatter(
            x='x', y='y', source=train_source, size=10, fill_color=color,
            fill_alpha=.9, line_color='#111111', line_alpha=.9, marker=marker)
        scatters['test_scatter_class_%s' % i] = fig.scatter(
            x='x', y='y', source=test_source, size=10, fill_color=color,
            fill_alpha=.3, line_color='#111111', line_alpha=.3, marker=marker)

    # Checkboxes to show/hide data points
    # -------------------------------------------------------------------------
    checkbox_show_points = CheckboxGroup(
        labels=["Show training data", "Show testing data"], active=[0, 1],
        callback=CustomJS(args=scatters, code="""
        var show_training_data = cb_obj.active.indexOf(0) !== -1;
        var show_testing_data = cb_obj.active.indexOf(1) !== -1;
        var max_classes_to_check = 20;
        var source_name_train, source_name_test;
        var source_train, source_test;
        for (var class_nr=0; class_nr < max_classes_to_check; ++class_nr) {
            try {
                source_name_train = "train_scatter_class_"+class_nr.toString();
                source_name_test = "test_scatter_class_"+class_nr.toString();
                source_train = eval(source_name_train);
                source_test = eval(source_name_test);
                source_train.visible = show_training_data;
                source_test.visible = show_testing_data;
            } catch (e) {
                // Do nothing.
                // We end up in here if eval() name is undefined.
            }
        }
    """))

    # Sliders
    # -------------------------------------------------------------------------
    slider1 = Slider(**classifiers.CLFS[classifiers.DEFAULT_CLF]['slider1'],
                     width=800)
    slider2 = Slider(**classifiers.CLFS[classifiers.DEFAULT_CLF]['slider2'],
                     width=800)

    # Classifier select box
    # -------------------------------------------------------------------------
    select_clf = Select(title='Classifier', value=classifiers.DEFAULT_CLF,
                        options=list(classifiers.CLFS.keys()))

    # Classifier select box callback
    # -------------------------------------------------------------------------
    select_clf.callback = CustomJS(
        args=dict(slider1=slider1, slider2=slider2), code="""
        var selected_clf = cb_obj.value;
        jQuery.ajax({
            type: 'POST',
            url: '/get_classifier_info',
            data: {'newly_selected_clf': selected_clf},
            dataType: 'json',
            success: function (json_from_server) {

                // Sliders have 'title' attribute but changing it
                // has no effect. Thus, change the label through jQuery.
                var slider1_label = $("label[for='" + slider1.id + "']");
                var slider2_label = $("label[for='" + slider2.id + "']");
                slider1_label.html(json_from_server.slider1.title);
                slider2_label.html(json_from_server.slider2.title);

                slider1.start = json_from_server.slider1.start;
                slider1.end = json_from_server.slider1.end;
                slider1.step = json_from_server.slider1.step;
                slider1.value = json_from_server.slider1.value;

                slider2.start = json_from_server.slider2.start;
                slider2.end = json_from_server.slider2.end;
                slider2.step = json_from_server.slider2.step;
                slider2.value = json_from_server.slider2.value;
            },
            error: function() {
                alert("Oh no, something went wrong.");
            }
        });
        """)

    # Slider callback
    # -------------------------------------------------------------------------
    general_slider_callback = CustomJS(
        args=dict(select_clf=select_clf, slider1=slider1, slider2=slider2,
                  source_region=source_region),
        code="""
        var clf_name = select_clf.value;
        var param1_name = $("label[for='" + slider1.id + "']").html();
        var param2_name = $("label[for='" + slider2.id + "']").html();
        var param1_val = slider1.value;
        var param2_val = slider2.value;
        jQuery.ajax({
            type: 'POST',
            url: '/get_new_decision_boundary',
            data: {'clf_name': clf_name,
                   'param1_name': param1_name, 'param1_val': param1_val,
                   'param2_name': param2_name, 'param2_val': param2_val},
            dataType: 'json',
            success: function (json_from_server) {
                source_region.data.d = [json_from_server.new_dec_bound];
                source_region.trigger('change');
                accuracy_text.text = "<h2 class='accuracy'>Test accuracy: " +
                                     json_from_server.accuracy + " %</h2>";
            },
            error: function() {
                alert("Oh no, something went wrong.");
            }
        });
        """)
    slider1.callback = general_slider_callback
    slider2.callback = general_slider_callback

    # Accuracy textbox
    # -------------------------------------------------------------------------
    accuracy_text = Div(text="", width=600, height=100)
    general_slider_callback.args["accuracy_text"] = accuracy_text

    # Generate layout, HTML, CSS and JS
    # -------------------------------------------------------------------------
    layout = column(row(select_clf, slider1),
                    row(checkbox_show_points, slider2),
                    row(fig),
                    row(accuracy_text))
    js_resources = INLINE.render_js()
    css_resources = INLINE.render_css()
    script, div = components(layout, INLINE)
    return script, div, js_resources, css_resources
