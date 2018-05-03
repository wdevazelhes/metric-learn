from abc import ABCMeta

import six
import sklearn
from numpy.linalg import inv, cholesky
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array


class BaseMetricLearner(BaseEstimator, TransformerMixin):
  def __init__(self):
    raise NotImplementedError('BaseMetricLearner should not be instantiated')

  def metric(self):
    """Computes the Mahalanobis matrix from the transformation matrix.

    .. math:: M = L^{\\top} L

    Returns
    -------
    M : (d x d) matrix
    """
    L = self.transformer()
    return L.T.dot(L)

  def transformer(self):
    """Computes the transformation matrix from the Mahalanobis matrix.

    L = cholesky(M).T

    Returns
    -------
    L : upper triangular (d x d) matrix
    """
    return cholesky(self.metric()).T

  def transform(self, X=None):
    """Applies the metric transformation.

    Parameters
    ----------
    X : (n x d) matrix, optional
        Data to transform. If not supplied, the training data will be used.

    Returns
    -------
    transformed : (n x d) matrix
        Input data transformed to the metric space by :math:`XL^{\\top}`
    """
    if X is None:
      X = self.X_
    else:
      X = check_array(X, accept_sparse=True)
    L = self.transformer()
    return X.dot(L.T)


class BaseMetricLearner:

  def score_pairs(self, X):
      """Compute similarities for each pair in X.

      If ``self`` has a ``preprocessors`` attribute that is not None,
      then elements in the pairs are transformed first, that is::

          X_0 = self.preprocessor_[0].transform(X[:, 0])
          X_1 = self.preprocessor_[1].transform(X[:, 1])
          X = np.hstack([X_0, X_1])

      Parameters
      ----------
      X : array-like, shape=(n_pairs, 2, n_features) or list
          Features for each pair of samples to score.

      Returns
      -------
      similarities : array-like, shape=(n_pairs,)
          Similarities are not normalized: they can take arbitrary
          values in the real domain, and 0 has no particular meaning.
          Higher similarity values mean "closer" pair elements in the
          metric space learned by the model.
      """



class BaseExplicitLearner(BaseMetricLearner):

  def embed(self, X, kind=None):
      """Embed datapoints in the metric space learned by the model.

      If the model has a ``preprocessor_`` attribute that is not None,
      then ``X`` is first transformed by calling
      ``X = self.preprocessor_[kind].transform(X)``.

      If ``preprocessor_`` is None or has a single element, kind can be
      left to its default None value.

      If there are several distinct preprocessors (when
      the model is fit on hetereogeneous pairs, triplets or quadruplets),
      then calling with ``kind=None`` raises a ``ValueError``.

      Parameters
      ----------
      X : array-like, shape=(n_samples, n_features)
          Features for each sample to transform.
      """

  # a default implementation of score_pairs can be provided such that:
  # score_pairs(np.hstack(X_0, X_1)) ==
  #             np.sum(embed(X_0, kind=0) * embed(X_1, kind=0), axis=1)


class BaseMetricClassifier(sklearn.base.BaseEstimator,
                           sklearn.base.ClassifierMixin,
                           sklearn.base.TransformerMixin,
                           MetricLearnerMixin):

  def __init__(self, classifier='1-nn', preprocessor=None):
      """Build a classifier using a metric-learning algorithm.

      Parameters
      ----------
      classifier : instance of ``sklearn.base.ClassifierMixin``
          Classifier fit on the training data transformed by the
          underlying metric learner to predict class labels from the
          training set.
          The default value ('1-nn') will use one-nearest neighbor
          classification.
          If set to `None`, the resulting model can only be used
          as a transformer: the `predict` and `decision_function` will
           raise `AttributeError`.

      preprocessor : instance sklearn transformers
          Transformer fit on the training data.


      Attributes
      ----------
      classifier_ : instance of ``sklearn.base.ClassifierMixin``
          Clone of the classifier passed as a constructor parameter
          fitted on the data transformed by the underlying metric
          learner.
      preprocessor_ : singleton list of instance transformer.
          Clone of the transformer passed as a constructor parameter
          fitted on the data transformed by the underlying metric
          learner.

      Returns
      -------
      self
      """


  def fit(self, X, y):
      """Fit model from class labeled data.

      Parameters
      ----------
      X : array-like, shape=(n_samples, n_features)
          Features for each sample in the training set.

      y : array-like
          y follows the multiclass or multilabel encodings admissible
          in scikit-learn.

      Returns
      -------
      self
      """

  def predict(self, X):
      """Predict class labels from the training set.
      """

  # todo: add transform that exposes publicly self.embed


class MetricTuplesClassifier:

  def fit(self, X):
    n_pairs = X.shape[0]
    assert X.shape[1] == 2
    preprocessor = self.preprocessor
    if hasattr(, '__array__'):
      preprocessor = ArrayIndexer(self.preprocessor)

    if preprocessor is not None:
      # ensure that cons params are not mutated by fit:
      preprocessor = clone(preprocessor)
      X_items = X.reshape(2 * n_pairs, -1)
      X_items = preprocessor.fit_transform(X_items)
      # todo: do the classification