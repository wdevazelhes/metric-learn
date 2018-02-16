import numpy as np
from scipy.sparse import csr_matrix

# TODO: make some utilities to check that a constrained dataset is coherent
# (ex: that there is no redundant pairs etc)

class ConstrainedDataset(object):

    def __init__(self, X, c):
        self.c = c
        self.X = X
        self.shape = (len(c), X.shape[1])


    def __getitem__(self, item):
        # Note that to avoid useless memory consumption, when splitting we
        # delete the points that are not used
        # TODO: deal with different types of slices (lists, arrays etc)
        c_sliced = self.c[item]
        unique_array = np.unique(c_sliced)
        inverted_index = self._build_inverted_index(unique_array)
        pruned_X = self.X[unique_array].copy()
        # copy so that the behaviour is always the  same
        rescaled_sliced_c = np.hstack([inverted_index[c_sliced[:, i]].A
                                       for i in range(c_sliced.shape[1])])
        return ConstrainedDataset(pruned_X, rescaled_sliced_c)

    def __len__(self):
        return self.shape

    def __str__(self):
        return self.asarray().__str__()

    def __repr__(self):
        return self.asarray().__repr__()

    def asarray(self):
        return np.stack(
            [self.X[self.c[:, i].ravel()] for i in range(self.c.shape[1])],
            axis=1)

    @staticmethod
    def _build_inverted_index(unique_array):
        inverted_index = csr_matrix((np.max(unique_array) + 1, 1), dtype=int)
        inverted_index[unique_array] = np.arange(len(unique_array))[:, None]
        return inverted_index

    @staticmethod
    def pairs_from_labels(y):
        # TODO: to be implemented
        return NotImplementedError

    @staticmethod
    def triplets_from_labels(y):
        # TODO: to be implemented
        return NotImplementedError