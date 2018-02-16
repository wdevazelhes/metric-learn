from .base import PairsSupervisedMetricLearner


class MahalanobisMetricLearning(PairsSupervisedMetricLearner):


    def __init__(self):
        super(MahalanobisMetricLearning, self).__init__()

    def fit(self, constrained_dataset, y):
        return NotImplementedError