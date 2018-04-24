import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from metric_learn import MetricClassifier
from sklearn.neighbors import KNeighborsClassifier


RNG_SEED = 0

iris = load_iris()
n_samples, n_features = iris.data.shape

X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target,
                                                    test_size=0.2,
                                                    random_state=RNG_SEED)

clf = MetricClassifier(metric_learner='lsml',
                       classifier=KNeighborsClassifier(),
                       n_samples=1000, with_replacement=True,
                       random_state=RNG_SEED)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print(classification_report(y_test, y_pred))

#
# Learned metric space visualization
#

pca_test = PCA(n_components=2).fit_transform(clf.transform(X_test))

fig, ax = plt.subplots(figsize=(10, 10))
ax.scatter(pca_test[:, 0], pca_test[:, 1], c=y_test, marker='x',
           label='test')
ax.legend(loc='best')
plt.show()
