import pytest
import numpy as np
from scipy.sparse.csgraph import laplacian
from sklearn.utils import check_random_state
from scipy.sparse import coo_matrix
from numpy.testing import assert_array_almost_equal

RNG = check_random_state(0)

def test_loss_sdml():
    n_samples = 10
    n_dims = 5
    X = RNG.randn(n_samples, n_dims)
    c = np.array([[1, 2], [3, 6], [2, 3], [4, 2], [5, 3]])
    n_pairs = len(c)
    pairs = X[c]
    y = RNG.choice([-1, 1], (n_pairs))


    adj = coo_matrix((y, (c[:, 0], c[:, 1])), shape=(n_samples,) * 2)
    adj_sym = adj + adj.T

    diff = pairs[:, 0] - pairs[:, 1]
    weighted_outer_prod = (diff.T * y).dot(diff)

    L = laplacian(adj_sym, normed=False)
    X.T.dot(L.dot(X))

    assert_array_almost_equal(weighted_outer_prod, X.T.dot(L.dot(X)))



def test_loss_sdml_with_duplicates():
    n_samples = 10
    n_dims = 5
    X = RNG.randn(n_samples, n_dims)
    c = np.array([[1, 2], [3, 6], [2, 3], [4, 2], [5, 3], [5, 3]])
    pairs = X[c]
    y = [-1, 1, 1, 1, 1, 1]


    adj = coo_matrix((y, (c[:, 0], c[:, 1])), shape=(n_samples,) * 2)
    adj_sym = adj + adj.T

    diff = pairs[:, 0] - pairs[:, 1]
    weighted_outer_prod = (diff.T * y).dot(diff)

    L = laplacian(adj_sym, normed=False)
    X.T.dot(L.dot(X))

    assert_array_almost_equal(weighted_outer_prod, X.T.dot(L.dot(X)))