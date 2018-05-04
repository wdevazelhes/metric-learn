"""
Information Theoretic Metric Learning, Kulis et al., ICML 2007

ITML minimizes the differential relative entropy between two multivariate
Gaussians under constraints on the distance function,
which can be formulated into a Bregman optimization problem by minimizing the
LogDet divergence subject to linear constraints.
This algorithm can handle a wide variety of constraints and can optionally
incorporate a prior on the distance function.
Unlike some other methods, ITML does not rely on an eigenvalue computation
or semi-definite programming.

Adapted from Matlab code at http://www.cs.utexas.edu/users/pjain/itml/
"""

from __future__ import print_function, absolute_import
import numpy as np
from six.moves import xrange
from sklearn.base import MetaEstimatorMixin, TransformerMixin
from sklearn.metrics import pairwise_distances
from sklearn.utils.validation import check_array, check_X_y

from .base_metric import BaseMetricLearner, MahalanobisMetricMixin, \
  MetricTuplesClassifier
from .constraints import Constraints
from ._util import vector_norm


class ITML(MetricTuplesClassifier, MahalanobisMetricMixin):
  """Information Theoretic Metric Learning (ITML)"""
  def __init__(self, *args, gamma=1., max_iter=1000,
               convergence_threshold=1e-3,
               A0=None, verbose=False, **kwargs):
    """Initialize ITML.

    Parameters
    ----------
    gamma : float, optional
        value for slack variables

    max_iter : int, optional

    convergence_threshold : float, optional

    A0 : (d x d) matrix, optional
        initial regularization matrix, defaults to identity

    verbose : bool, optional
        if True, prints information while learning
    """
    self.gamma = gamma
    self.max_iter = max_iter
    self.convergence_threshold = convergence_threshold
    self.A0 = A0
    self.verbose = verbose
    super(ITML, self).__init__()

  def _process_pairs(self, pairs, y, bounds):
    y = y.astype(bool).ravel()  # todo: make cleaner implem
    pairs = check_array(pairs, accept_sparse=False,
                                      ensure_2d=False, allow_nd=True)

    # check to make sure that no two constrained vectors are identical
    pos_pairs, neg_pairs = pairs[y], pairs[~y]
    pos_no_ident = vector_norm(pos_pairs[:, 0, :] - pos_pairs[:, 1, :]) > 1e-9
    pos_pairs = pos_pairs[pos_no_ident]
    neg_no_ident = vector_norm(neg_pairs[:, 0, :] - neg_pairs[:, 1, :]) > 1e-9
    neg_pairs = neg_pairs[neg_no_ident]
    # init bounds
    if bounds is None:
      self.bounds_ = np.percentile(pairwise_distances(pairs[:, 0, :],
                                                      pairs[:, 1, :]),
                                   (5, 95))
    else:
      assert len(bounds) == 2
      self.bounds_ = bounds
    self.bounds_[self.bounds_==0] = 1e-9
    # init metric
    if self.A0 is None:
      self.A_ = np.identity(pairs.shape[2])
    else:
      self.A_ = check_array(self.A0)
    pairs = np.vstack([pos_pairs, neg_pairs])
    y = np.hstack([np.ones(len(pos_pairs)), np.zeros(len(neg_pairs))])
    y = y.astype(bool)
    return pairs, y


  def fit(self, y, pairs, bounds=None):
    """Learn the ITML model.

    Parameters
    ----------
    X : (n x d) data matrix
        each row corresponds to a single instance
    constraints : 4-tuple of arrays
        (a,b,c,d) indices into X, with (a,b) specifying positive and (c,d)
        negative pairs
    bounds : list (pos,neg) pairs, optional
        bounds on similarity, s.t. d(X[a],X[b]) < pos and d(X[c],X[d]) > neg
    """
    pairs, y = self._process_pairs(pairs, y, bounds)
    gamma = self.gamma
    pos_pairs, neg_pairs = pairs[y], pairs[~y]
    num_pos = len(pos_pairs)
    num_neg = len(neg_pairs)
    _lambda = np.zeros(num_pos + num_neg)
    lambdaold = np.zeros_like(_lambda)
    gamma_proj = 1. if gamma is np.inf else gamma/(gamma+1.)
    pos_bhat = np.zeros(num_pos) + self.bounds_[0]
    neg_bhat = np.zeros(num_neg) + self.bounds_[1]
    pos_vv = pos_pairs[:, 0, :] - pos_pairs[:, 1, :]
    neg_vv = neg_pairs[:, 0, :] - neg_pairs[:, 1, :]
    A = self.A_

    for it in xrange(self.max_iter):
      # update positives
      for i,v in enumerate(pos_vv):
        wtw = v.dot(A).dot(v)  # scalar
        alpha = min(_lambda[i], gamma_proj*(1./wtw - 1./pos_bhat[i]))
        _lambda[i] -= alpha
        beta = alpha/(1 - alpha*wtw)
        pos_bhat[i] = 1./((1 / pos_bhat[i]) + (alpha / gamma))
        Av = A.dot(v)
        A += np.outer(Av, Av * beta)

      # update negatives
      for i,v in enumerate(neg_vv):
        wtw = v.dot(A).dot(v)  # scalar
        alpha = min(_lambda[i+num_pos], gamma_proj*(1./neg_bhat[i] - 1./wtw))
        _lambda[i+num_pos] -= alpha
        beta = -alpha/(1 + alpha*wtw)
        neg_bhat[i] = 1./((1 / neg_bhat[i]) - (alpha / gamma))
        Av = A.dot(v)
        A += np.outer(Av, Av * beta)

      normsum = np.linalg.norm(_lambda) + np.linalg.norm(lambdaold)
      if normsum == 0:
        conv = np.inf
        break
      conv = np.abs(lambdaold - _lambda).sum() / normsum
      if conv < self.convergence_threshold:
        break
      lambdaold = _lambda.copy()
      if self.verbose:
        print('itml iter: %d, conv = %f' % (it, conv))

    if self.verbose:
      print('itml converged at iter: %d, conv = %f' % (it, conv))
    self.n_iter_ = it
    return self


  def metric(self):
    return self.A_



class ITMLTransformer(TransformerMixin, MetaEstimatorMixin):
  """Information Theoretic Metric Learning (ITML)"""
  def __init__(self, gamma=1., max_iter=1000, convergence_threshold=1e-3,
               num_labeled=np.inf, num_constraints=None, bounds=None, A0=None,
               verbose=False):
    """Initialize the learner.

    Parameters
    ----------
    gamma : float, optional
        value for slack variables
    max_iter : int, optional
    convergence_threshold : float, optional
    num_labeled : int, optional
        number of labels to preserve for training
    num_constraints: int, optional
        number of constraints to generate
    bounds : list (pos,neg) pairs, optional
        bounds on similarity, s.t. d(X[a],X[b]) < pos and d(X[c],X[d]) > neg
    A0 : (d x d) matrix, optional
        initial regularization matrix, defaults to identity
    verbose : bool, optional
        if True, prints information while learning
    """
    self.gamma=gamma
    self.max_iter=max_iter
    self.convergence_threshold=convergence_threshold
    self.A0=A0
    self.verbose=verbose

    self.num_labeled = num_labeled
    self.num_constraints = num_constraints
    self.bounds = bounds

  def fit(self, X, y, random_state=np.random):
    """Create constraints from labels and learn the ITML model.

    Parameters
    ----------
    X : (n x d) matrix
        Input data, where each row corresponds to a single instance.

    y : (n) array-like
        Data labels.

    random_state : numpy.random.RandomState, optional
        If provided, controls random number generation.
    """

    self.itml = ITML(self.gamma,
                     self.max_iter,
                     self.convergence_threshold,
                     self.A0,
                     self.verbose)
    X, y = check_X_y(X, y)
    num_constraints = self.num_constraints
    if num_constraints is None:
      num_classes = len(np.unique(y))
      num_constraints = 20 * num_classes**2

    c = Constraints.random_subset(y, self.num_labeled,
                                  random_state=random_state)
    pos_neg = c.positive_negative_pairs(num_constraints,
                                        random_state=random_state)
    return self.itml.fit(self, X, pos_neg, bounds=self.bounds)

  def transform(self, X):
    return self.itml.embed(X)
