from ..base import WeaklySupervisedMetricLearner
from sklearn.metrics import roc_auc_score

class PairsSupervisedMetricLearner(WeaklySupervisedMetricLearner):


    def fit(self, constrained_dataset, y):
        return NotImplementedError
    # TODO: maybe instead inherit from SupervisedMixin


    def predict(self, constrained_dataset):
        return self.get_metric()(constrained_dataset.X[constrained_dataset.c[
                                                       :, 0]],
                                 constrained_dataset.X[constrained_dataset.c[
                                                       :, 1]])

    def score(self, constrained_dataset, y):
        # TODO: use decision function or predict ?
        return roc_auc_score(y, self.decision_function(constrained_dataset))