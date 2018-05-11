import pytest
import numpy as np
from scipy.sparse.csgraph import laplacian
from sklearn.utils import check_random_state
from scipy.sparse import coo_matrix

RNG = check_random_state(0)

def test_loss_sdml():
    n_samples = 10
    n_dims = 5
    n_pairs = 20
    X = RNG.randn(n_samples, n_dims)
    c = RNG.randint(0, n_samples, (n_pairs, 2))
    c = np.array([[1, 2], [3, 6], [2, 3], [4, 2], [5, 3]])
    n_pairs = len(c)
    pairs = X[c]
    y = RNG.choice([-1, 1], (n_pairs))
    dot_prod = pairs[:, 0, :].T.dot(pairs[:, 1, :]) + \
               pairs[:, 1, :].T.dot(pairs[:, 0, :])
    diagonal = pairs[:, 0, :].T.dot(pairs[:, 0, :]) + \
               pairs[:, 1, :].T.dot(pairs[:, 1, :])
    # dot_prod[np.diag_indices(dot_prod)] = diagonal



    #
    # np.sum(pairs.T.dot(pairs), axis=1)
    # L = laplacian(W, normed=False)  # todo: to finish
    # return X.T.dot(L.dot(X))


    adj = coo_matrix((y, (c[:, 0], c[:, 1])), shape=(n_samples,) * 2)
    adj_sym = adj + adj.T

    W, D = laplacian(adj_sym, normed=False, return_diag=True)
    assert X.T.dot((D * np.eye(len(X))).dot(X)) == (pairs[:, 0].T * y).dot(
        pairs[:,0]) + \
           (pairs[:, 1].T * y).dot(pairs[:, 1])

    total_sum = \
    (pairs[:, 0].T * y).dot(pairs[:, 0]) + \
    (pairs[:, 1].T * y).dot(pairs[:, 1]) - \
    (pairs[:, 0].T * y).dot(pairs[:, 1]) - \
    (pairs[:, 1].T * y).dot(pairs[:, 0])

    diff = pairs[:, 0] - pairs[:, 1]
    sum_test = (diff.T * y).dot(diff)

    L = laplacian(adj_sym, normed=False)
    X.T.dot(L.dot(X))

