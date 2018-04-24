"""
Basic metric-learn example: learning on pairs of data
=====================================================

A simple example showing how to load dataset of pairs and how to train and
predict on them.
"""

#########################################################################
import numpy as np
from skimage.feature import hog
from sklearn.datasets import fetch_lfw_pairs
from sklearn.metrics import roc_auc_score
from sklearn.externals.joblib import Parallel, delayed
from sklearn.base import BaseEstimator, TransformerMixin
from metric_learn import MMCPairClassifier


data_train = fetch_lfw_pairs(subset='train', resize=0.5)
data_test = fetch_lfw_pairs(subset='test', resize=0.5)

print(data_train.pairs.shape)
print(data_test.pairs.shape)
subset_size = 500  # to speed up the example


class HogTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, n_jobs=1):
        self.n_jobs = n_jobs

    def transform(self, images):
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(hog)(img) for img in images)
        return np.array(results)


# Use an internal train/validation to find a threshold that matches a given
# precision goal:

pair_clf = MMCPairClassifier(cv=0.2, target_precision=0.8,
                             preprocessor=HogTransformer(n_jobs=4))

# or alternatively give a predifined similarity threshold for prediction:

pair_clf = MMCPairClassifier(threshold=0.6, similarity="cosine",
                             preprocessor=HogTransformer(n_jobs=4))

pair_clf.fit(data_train.pairs[:subset_size], data_train.target[:subset_size])

# Compute a pair classification score on soft-predictions:

predited_pair_scores = pair_clf.decision_function(data_test.pairs)
print("ROC AUC on pair classification: %0.2f"
      % roc_auc_score(data_test.target, predited_pair_scores))


# Visualize some hard predictions with the threshold found by Precision/Recall
# curve on validation split.

# TODO: imshow on pair_clf.predict outputs on the test set.