import unittest
import numpy as np
from sklearn.datasets import load_iris
from numpy.testing import assert_array_almost_equal

from metric_learn import (
    LMNN, NCA, LFDA, Covariance, MLKR,
    LSMLTransformer, ITMLTransformer, SDMLTransformer, RCATransformer, MMCTransformer)


class TestFitTransform(unittest.TestCase):
  @classmethod
  def setUpClass(self):
    # runs once per test class
    iris_data = load_iris()
    self.X = iris_data['data']
    self.y = iris_data['target']

  def test_cov(self):
    cov = Covariance()
    cov.fit(self.X)
    res_1 = cov.transform()

    cov = Covariance()
    res_2 = cov.fit_transform(self.X)
    # deterministic result
    assert_array_almost_equal(res_1, res_2)

  def test_lsmlTransformer(self):
    seed = np.random.RandomState(1234)
    lsml = LSMLTransformer(num_constraints=200)
    lsml.fit(self.X, self.y, random_state=seed)
    res_1 = lsml.transform()

    seed = np.random.RandomState(1234)
    lsml = LSMLTransformer(num_constraints=200)
    res_2 = lsml.fit_transform(self.X, self.y, random_state=seed)

    assert_array_almost_equal(res_1, res_2)

  def test_itmlTransformer(self):
    seed = np.random.RandomState(1234)
    itml = ITMLTransformer(num_constraints=200)
    itml.fit(self.X, self.y, random_state=seed)
    res_1 = itml.transform()

    seed = np.random.RandomState(1234)
    itml = ITMLTransformer(num_constraints=200)
    res_2 = itml.fit_transform(self.X, self.y, random_state=seed)

    assert_array_almost_equal(res_1, res_2)

  def test_lmnn(self):
    lmnn = LMNN(k=5, learn_rate=1e-6, verbose=False)
    lmnn.fit(self.X, self.y)
    res_1 = lmnn.transform()

    lmnn = LMNN(k=5, learn_rate=1e-6, verbose=False)
    res_2 = lmnn.fit_transform(self.X, self.y)

    assert_array_almost_equal(res_1, res_2)

  def test_sdmlTransformer(self):
    seed = np.random.RandomState(1234)
    sdml = SDMLTransformer(num_constraints=1500)
    sdml.fit(self.X, self.y, random_state=seed)
    res_1 = sdml.transform()

    seed = np.random.RandomState(1234)
    sdml = SDMLTransformer(num_constraints=1500)
    res_2 = sdml.fit_transform(self.X, self.y, random_state=seed)

    assert_array_almost_equal(res_1, res_2)

  def test_nca(self):
    n = self.X.shape[0]
    nca = NCA(max_iter=(100000//n), learning_rate=0.01)
    nca.fit(self.X, self.y)
    res_1 = nca.transform()

    nca = NCA(max_iter=(100000//n), learning_rate=0.01)
    res_2 = nca.fit_transform(self.X, self.y)

    assert_array_almost_equal(res_1, res_2)

  def test_lfda(self):
    lfda = LFDA(k=2, num_dims=2)
    lfda.fit(self.X, self.y)
    res_1 = lfda.transform()

    lfda = LFDA(k=2, num_dims=2)
    res_2 = lfda.fit_transform(self.X, self.y)

    # signs may be flipped, that's okay
    if np.sign(res_1[0,0]) != np.sign(res_2[0,0]):
        res_2 *= -1
    assert_array_almost_equal(res_1, res_2)

  def test_rcaTransformer(self):
    seed = np.random.RandomState(1234)
    rca = RCATransformer(num_dims=2, num_chunks=30, chunk_size=2)
    rca.fit(self.X, self.y, random_state=seed)
    res_1 = rca.transform()

    seed = np.random.RandomState(1234)
    rca = RCATransformer(num_dims=2, num_chunks=30, chunk_size=2)
    res_2 = rca.fit_transform(self.X, self.y, random_state=seed)

    assert_array_almost_equal(res_1, res_2)

  def test_mlkr(self):
    mlkr = MLKR(num_dims=2)
    mlkr.fit(self.X, self.y)
    res_1 = mlkr.transform()

    mlkr = MLKR(num_dims=2)
    res_2 = mlkr.fit_transform(self.X, self.y)

    assert_array_almost_equal(res_1, res_2)

  def test_mmcTransformer(self):
    seed = np.random.RandomState(1234)
    mmc = MMCTransformer(num_constraints=200)
    mmc.fit(self.X, self.y, random_state=seed)
    res_1 = mmc.transform(self.X)

    seed = np.random.RandomState(1234)
    mmc = MMCTransformer(num_constraints=200)
    res_2 = mmc.fit_transform(self.X, self.y, random_state=seed)

    assert_array_almost_equal(res_1, res_2)


if __name__ == '__main__':
  unittest.main()
