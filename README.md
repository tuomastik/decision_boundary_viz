# decision_boundary_viz
[![FOSSA Status](https://app.fossa.io/api/projects/git%2Bgithub.com%2Ftuomastik%2Fdecision_boundary_viz.svg?type=shield)](https://app.fossa.io/projects/git%2Bgithub.com%2Ftuomastik%2Fdecision_boundary_viz?ref=badge_shield)


A web app for visualizing decision boundaries of classifiers.

Supports 2-dimensional data and the following classifiers and their parameters:

* k-Nearest Neighbors
    * Number of neighbors
    * Weight function
        * Uniform weights
        * Points weighted by the inverse of their distance
* Random forest
    * The number of trees in the forest
    * The maximum depth of trees
* Support vector machine (RBF kernel)
    * Cost (C)
    * Gamma

## Demo

![](app_usage.gif)

## Dependencies

* [NumPy](http://www.numpy.org/) (numerical operations)
* [scikit-learn](http://scikit-learn.org/) (classifiers)
* [Bokeh](http://bokeh.pydata.org/) (interactive visualizations)
* [Flask](http://flask.pocoo.org/) (web framework)

```
conda create -n decision_boundary_viz python=3.5 numpy=1.11 scikit-learn=0.18 bokeh=0.12.4 flask=0.12
```


## License
[![FOSSA Status](https://app.fossa.io/api/projects/git%2Bgithub.com%2Ftuomastik%2Fdecision_boundary_viz.svg?type=large)](https://app.fossa.io/projects/git%2Bgithub.com%2Ftuomastik%2Fdecision_boundary_viz?ref=badge_large)