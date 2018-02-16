from ..base import WeaklySupervisedMetricLearner
import numpy as np

class TripletsSupervisedMetricLearner(WeaklySupervisedMetricLearner):


    def fit(self, constrained_dataset, y=None):
        return NotImplementedError
    # TODO: maybe instead inherit from UnsupervisedMixin


    def predict(self, constrained_dataset):
        return self.get_metric()(constrained_dataset.X[constrained_dataset.c[
                                                       :, 0]],
                                 constrained_dataset.X[constrained_dataset.c[
                                                       :, 1]]) - \
               self.get_metric()(constrained_dataset.X[constrained_dataset.c[
                                                       :, 0]],
                                 constrained_dataset.X[constrained_dataset.c[
                                                       :, 2]])

    def  score(self, constrained_dataset, y=None):
        # TODO: use decision function or predict ?
        predicted_sign = self.decision_function(constrained_dataset) < 0
        return np.sum(predicted_sign) / predicted_sign.shape[0]
