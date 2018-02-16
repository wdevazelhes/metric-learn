import numpy as np
from sklearn.base import BaseEstimator

class MetricLearnerMixin(BaseEstimator):

    def __init__(self):
        self._components = None

    def fit(self, constrained_dataset, y):
        return NotImplementedError

    def _transform(self, X):
        return NotImplementedError

    def get_metric(self):
        A = self._components.copy()
        def learned_pairwise_distances(X, Y):
            pairwise_diffs = X - Y
            return np.sqrt(np.sum(pairwise_diffs.dot(A) * pairwise_diffs,
                                  axis=1))
            # TODO:  the same as in scikit learn : if Y is None, then do pairwise
            # distances of X
            # TODO: look at the developed version in
            # sklearn.metrics.pairwise.euclidean_distances
        return learned_pairwise_distances


