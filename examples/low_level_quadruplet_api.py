import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.decomposition  import PCA
from metric_learn.sampling import sample_quadruplets
from metric_learn.sampling import sample_pairs
from metric_learn import LSML


RNG_SEED = 0

iris = load_iris()
n_samples, n_features = iris.data.shape

idx_train, idx_test, y_train, y_test = train_test_split(
    np.arange(n_samples), iris.target, test_size=0.2, random_state=RNG_SEED)

#
# Model fitting
#

quadruplets_train = sample_quadruplets(n_samples=1000, idx_train, y_train,
                                       random_state=RNG_SEED)

metric_model = LSML(data_fetcher=iris.data).fit_quadruplets(quadruplets_train)

#
# Quantitative model evaluation
#

pair_indices, pair_labels = sample_pairs(idx_test, y_test,
                                         n_samples=1000,
                                         neg_ratio=0.666,
                                         with_replacement=True,
                                         strategy='classwise_uniform',
                                         random_state=RNG_SEED)


predicted_pair_scores = metric_model.score_pairs(pair_indices)
print(roc_auc_score(pair_labels, predicted_pair_scores))

#
# Learned metric space visualization
#

pca = PCA(n_components=2)
pca_train = pca.fit_transform(metric_model.transform(iris.data[idx_train]))
pca_test = pca.transform(metric_model.transform(iris.data[idx_test]))

fig, ax = plt.subplots(figsize=(10, 10))
ax.scatter(pca_train[:, 0], pca_train[:, 1], c=y_train, marker='o',
           label='train')
ax.scatter(pca_test[:, 0], pca_test[:, 1], c=y_test, marker='x',
           label='test')
ax.legend(loc='best')
plt.show()

