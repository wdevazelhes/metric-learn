from ..base import MetricLearnerMixin
from sklearn.base import TransformerMixin

class SupervisedMetricLearner(MetricLearnerMixin, TransformerMixin):

    def fit(self, X, y):
        return NotImplementedError

    def transform(self, X):
        return super(SupervisedMetricLearner, self)._transform(X)
