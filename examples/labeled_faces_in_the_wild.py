"""
Basic metric-learn example: learning on pairs of data
=====================================================

A simple example showing how to load dataset of pairs and how to train and
predict on them.
"""

#########################################################################

from sklearn.datasets import fetch_lfw_pairs

from metric_learn import MMC
from sklearn.metrics import classification_report


data_train = fetch_lfw_pairs(subset='train')
data_test = fetch_lfw_pairs(subset='test')

print(data_train.pairs.shape)
print(data_test.pairs.shape)

model = PairsClassifier()

model.fit(data_train.pairs, data_train.target)

predited_pair_scores = model.decision_function(data_test.pairs)
print(classification_report(data_test.target, predited_pair_scores))