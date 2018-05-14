"""
Qi et al.
An efficient sparse metric learning in high-dimensional space via
L1-penalized log-determinant regularization.
ICML 2009

Adapted from https://gist.github.com/kcarnold/5439945
Paper: http://lms.comp.nus.edu.sg/sites/default/files/publication-attachments/icml09-guojun.pdf
"""

from __future__ import absolute_import
import numpy as np
from scipy.sparse.csgraph import laplacian
from sklearn.base import MetaEstimatorMixin, TransformerMixin
from sklearn.covariance import graph_lasso
from sklearn.utils.extmath import pinvh
from sklearn.utils.validation import check_array

from .base_metric import BaseMetricLearner, MetricTuplesClassifier, \
  MahalanobisMetricMixin
from .constraints import Constraints, wrap_pairs


class SDML(MetricTuplesClassifier, MahalanobisMetricMixin):
  def __init__(self, balance_param=0.5, sparsity_param=0.01, use_cov=True,
               verbose=False):
    """
    Parameters
    ----------
    balance_param : float, optional
        trade off between sparsity and M0 prior

    sparsity_param : float, optional
        trade off between optimizer and sparseness (see graph_lasso)

    use_cov : bool, optional
        controls prior matrix, will use the identity if use_cov=False

    verbose : bool, optional
        if True, prints information while learning
    """
    self.balance_param = balance_param
    self.sparsity_param = sparsity_param
    self.use_cov = use_cov
    self.verbose = verbose
    super(SDML, self).__init__()

  def _prepare_pairs(self, pairs, y):
    pairs = check_array(pairs, accept_sparse=False,
                                      ensure_2d=False, allow_nd=True)
    # set up prior M
    if self.use_cov:
      X = np.unique(pairs.reshape(-1, pairs.shape[2]), axis=0)
      self.M_ = pinvh(np.cov(X, rowvar = False))
    else:
      self.M_ = np.identity(pairs.shape[2])
    diff = pairs[:, 0] - pairs[:, 1]
    return (diff.T * y).dot(diff)

  def metric(self):
    return self.M_

  def fit(self, pairs, y):
    """Learn the SDML model.

    Parameters
    ----------
    X : array-like, shape (n, d)
        data matrix, where each row corresponds to a single instance
    W : array-like, shape (n, n)
        connectivity graph, with +1 for positive pairs and -1 for negative

    Returns
    -------
    self : object
        Returns the instance.
    """
    loss_matrix = self._prepare_pairs(pairs, y)
    P = self.M_ + self.balance_param * loss_matrix
    emp_cov = pinvh(P)
    # hack: ensure positive semidefinite
    emp_cov = emp_cov.T.dot(emp_cov)
    _, self.M_ = graph_lasso(emp_cov, self.sparsity_param, verbose=self.verbose)
    return self


class SDMLTransformer(TransformerMixin, MetaEstimatorMixin):
  def __init__(self, balance_param=0.5, sparsity_param=0.01, use_cov=True,
               num_labeled=np.inf, num_constraints=None, verbose=False):
    """
    Parameters
    ----------
    balance_param : float, optional
        trade off between sparsity and M0 prior
    sparsity_param : float, optional
        trade off between optimizer and sparseness (see graph_lasso)
    use_cov : bool, optional
        controls prior matrix, will use the identity if use_cov=False
    num_labeled : int, optional
        number of labels to preserve for training
    num_constraints : int, optional
        number of constraints to generate
    verbose : bool, optional
        if True, prints information while learning
    """
    self.balance_param=balance_param
    self.sparsity_param=sparsity_param
    self.use_cov=use_cov
    self.verbose=verbose

    self.num_labeled = num_labeled
    self.num_constraints = num_constraints

  def fit(self, X, y, random_state=np.random):
    """Create constraints from labels and learn the SDML model.

    Parameters
    ----------
    X : array-like, shape (n, d)
        data matrix, where each row corresponds to a single instance
    y : array-like, shape (n,)
        data labels, one for each instance
    random_state : {numpy.random.RandomState, int}, optional
        Random number generator or random seed. If not given, the singleton
        numpy.random will be used.

    Returns
    -------
    self : object
        Returns the instance.
    """
    self.sdml = SDML(balance_param = self.balance_param,
                     sparsity_param = self.sparsity_param,
                     use_cov = self.use_cov,
                     verbose = self.verbose)
    y = check_array(y, ensure_2d=False)
    num_constraints = self.num_constraints
    if num_constraints is None:
      num_classes = len(np.unique(y))
      num_constraints = 20 * num_classes**2

    c = Constraints.random_subset(y, self.num_labeled,
                                  random_state=random_state)
    pos_neg = c.positive_negative_pairs(num_constraints,
                                              random_state=random_state)
    pairs, y = wrap_pairs(X, pos_neg)
    return self.sdml.fit(self, pairs, y)
