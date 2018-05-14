Mahalanobis Metric Learning for Clustering (MMC)
================================================

.. automodule:: metric_learn.mmc
    :members:
    :undoc-members:
    :inherited-members:
    :show-inheritance:

Example Code
------------

::

    from metric_learn import MMCTransformer
    from sklearn.datasets import load_iris

    iris_data = load_iris()
    X = iris_data['data']
    Y = iris_data['target']

    mmc = MMCTransformer(num_constraints=200)
    mmc.fit(X, Y)

References
----------
`Distance metric learning with application to clustering with side-information <http://papers.nips.cc/paper/2164-distance-metric-learning-with-application-to-clustering-with-side-information.pdf>`_ Xing, Jordan, Russell, Ng.
