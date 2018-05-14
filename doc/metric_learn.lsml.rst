Least Squares Metric Learning (LSML)
====================================

.. automodule:: metric_learn.lsml
    :members:
    :undoc-members:
    :inherited-members:
    :show-inheritance:

Example Code
------------

::

    from metric_learn import LSMLTransformer
    from sklearn.datasets import load_iris

    iris_data = load_iris()
    X = iris_data['data']
    Y = iris_data['target']

    lsml = LSMLTransformer(num_constraints=200)
    lsml.fit(X, Y)

References
----------

