from ..base import MetricLearnerMixin

class WeaklySupervisedMetricLearner(MetricLearnerMixin):


    def fit(self, constrained_dataset, y):
        return NotImplementedError

    def transform(self, constrained_dataset):
        return super(WeaklySupervisedMetricLearner, self)._transform(
            constrained_dataset.X)

    def decision_function(self, constrained_dataset):
        return self.predict(constrained_dataset)

    def predict(self, constrained_dataset):
        return NotImplementedError

    def score(self, constrained_dataset, y):
        return NotImplementedError