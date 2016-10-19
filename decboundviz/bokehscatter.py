# -*- coding: utf-8 -*-

from bokeh.plotting import figure
from bokeh.models import (ColumnDataSource, CustomJS, Select, Slider,
                          HoverTool)
from bokeh.embed import components
from bokeh.resources import INLINE
from bokeh.layouts import column, row

from . import decisionboundary


CLASSIFIERS = {
    'Random forest': {
        'slider1': {'title': 'Number of trees', 'start': 0, 'end': 10,
                    'step': 1, 'value': 4},
        'slider2': {'title': 'Max depth', 'start': 0, 'end': 10,
                    'step': 1, 'value': 4},
    },
    'k-NN': {
        'slider1': {'title': 'n_neighbors', 'start': 1, 'end': 30,
                    'step': 1, 'value': 4},  # 'Nr of neighbors',
        'slider2': {'title': 'weights', 'start': 0, 'end': 1,
                    'step': 1, 'value': 1},  # 'Weight function'
    },
    'SVM': {
        'slider1': {'title': 'C', 'start': 0, 'end': 10,
                    'step': 1, 'value': 4},
        'slider2': {'title': 'Gamma', 'start': 0, 'end': 10,
                    'step': 1, 'value': 4},
    }
}
DEFAULT_CLF = 'k-NN'


def plot_decision_region(fig):
    d, x_min, y_min, dw, dh = decisionboundary.knn()
    source_region = ColumnDataSource(data={'d': [d]})
    region = fig.image(image='d', x=x_min, y=y_min,
                       dw=dw, dh=dh, source=source_region,
                       palette=['#DC9C76', '#D6655A'])
    return region, source_region


def plot_points(fig, source, classes):
    colors = []
    for class_nr in classes:
        if class_nr == 0:
            colors.append("#DC9C76")
        else:
            colors.append("#D6655A")

    return fig.scatter(x='x', y='y', source=source, size=10, fill_color=colors,
                       fill_alpha=0.9, line_color='#000000')


def create():

    x_train = decisionboundary.X_TRAIN
    y_train = decisionboundary.Y_TRAIN
    source = ColumnDataSource(data=dict(x=x_train[:, 0], y=x_train[:, 1]))

    # Figure
    # -------------------------------------------------------------------------
    tools = [
        HoverTool(tooltips=[("Index", "$index"), ("X", "@x"), ("Y", "@y")]),
        "pan, wheel_zoom, box_select, lasso_select, reset, save"]

    fig = figure(tools=tools, plot_width=1110, plot_height=600,
                 x_range=decisionboundary.get_padded_range(x_train[:, 0]),
                 y_range=decisionboundary.get_padded_range(x_train[:, 1]),
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
    plot_points(fig, source, y_train)

    # Sliders
    # -------------------------------------------------------------------------
    slider1 = Slider(**CLASSIFIERS[DEFAULT_CLF]['slider1'])
    slider2 = Slider(**CLASSIFIERS[DEFAULT_CLF]['slider2'])

    # Classifier select box
    # -------------------------------------------------------------------------
    select_clf = Select(title='Classifier', value=DEFAULT_CLF,
                        options=list(CLASSIFIERS.keys()))

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

                // slider1.trigger('change');
                // slider2.trigger('change');
            },
            error: function() {
                alert("Oh no, something went wrong.");
            }
        });
        """)

    # Slider callback
    # -------------------------------------------------------------------------
    general_slider_callback = CustomJS(
        args=dict(source=source, select_clf=select_clf, slider1=slider1,
                  slider2=slider2, region=region, source_region=source_region),
        code="""
        var clf_name = select_clf.value;
        var attr1_name = $("label[for='" + slider1.id + "']").html();
        var attr2_name = $("label[for='" + slider2.id + "']").html();
        var attr1_val = slider1.value;
        var attr2_val = slider2.value;
        jQuery.ajax({
            type: 'POST',
            url: '/get_new_decision_boundary',
            data: {'clf_name': clf_name,
                   'attr1_name': attr1_name, 'attr1_val': attr1_val,
                   'attr2_name': attr2_name, 'attr2_val': attr2_val},
            dataType: 'json',
            success: function (json_from_server) {
                new_surface = json_from_server[0];
                source_region.data.d = [new_surface];
                source_region.trigger('change');
            },
            error: function() {
                alert("Oh no, something went wrong.");
            }
        });
        """)
    slider1.callback = general_slider_callback
    slider2.callback = general_slider_callback

    # Generate layout, HTML, CSS and JS
    # -------------------------------------------------------------------------
    layout = column(row(select_clf, slider1, slider2),
                    row(fig))
    js_resources = INLINE.render_js()
    css_resources = INLINE.render_css()
    script, div = components(layout, INLINE)
    return script, div, js_resources, css_resources
