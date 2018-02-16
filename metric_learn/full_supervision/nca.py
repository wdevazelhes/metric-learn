from .base import SupervisedMetricLearner

class NeighborhoodComponentsAnalysis(SupervisedMetricLearner):


    def __init__(self):
        super(NeighborhoodComponentsAnalysis, self).__init__()

    def fit(self, constrained_dataset, y):
        return NotImplementedError