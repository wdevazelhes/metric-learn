import unittest
from sklearn import clone
from sklearn.cluster import KMeans
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.utils.estimator_checks import is_public_parameter, check_estimator
from sklearn.utils.testing import set_random_state, assert_true, \
    assert_allclose_dense_sparse, assert_dict_equal, assert_false

from metric_learn import ITML, LSML, MMC, SDML
from metric_learn.constraints import ConstrainedDataset
from sklearn.utils import check_random_state, check_X_y, check_array
import numpy as np

num_points = 100
num_features = 5
num_constraints = 100

RNG = check_random_state(0)

X = RNG.randn(num_points, num_features)
y = RNG.randint(0, 2, num_constraints)
group = RNG.randint(0, 3, num_constraints)


class _TestWeaklySupervisedBase(unittest.TestCase):

    def setUp(self):
        self.c = RNG.randint(0, num_points, (num_constraints,
                                             self.num_points_in_constraint))
        self.X_constrained = ConstrainedDataset(X, self.c)
        self.X_constrained_train, self.X_constrained_test, self.y_train, \
        self.y_test = train_test_split(self.X_constrained, y)
        set_random_state(self.estimator) # sets the algorithm random seed (if
        #  any)

    def test_cross_validation(self):
        # test that you can do cross validation on a ConstrainedDataset with
        #  a WeaklySupervisedMetricLearner
        estimator = clone(self.estimator)
        self.assertTrue(np.isfinite(cross_val_score(estimator,
                                    self.X_constrained, y)).all())

    def check_score(self, estimator, X_constrained, y):
        score = estimator.score(X_constrained, y)
        self.assertTrue(np.isfinite(score))

    def check_predict(self, estimator, X_constrained):
        y_predicted = estimator.predict(X_constrained)
        self.assertEqual(len(y_predicted), len(X_constrained))

    def check_transform(self, estimator, X_constrained):
        X_transformed = estimator.transform(X_constrained)
        self.assertEqual(len(X_transformed), len(X_constrained.X))

    def test_simple_estimator(self):
        estimator = clone(self.estimator)
        estimator.fit(self.X_constrained_train, self.y_train)
        self.check_score(estimator, self.X_constrained_test, self.y_test)
        self.check_predict(estimator, self.X_constrained_test)
        self.check_transform(estimator, self.X_constrained_test)

    def test_pipelining_with_transformer(self):
        """
        Test that weakly supervised estimators fit well into pipelines
        """
        # test in a pipeline with KMeans
        estimator = clone(self.estimator)
        pipe = make_pipeline(estimator, KMeans())
        pipe.fit(self.X_constrained_train, self.y_train)
        self.check_score(pipe, self.X_constrained_test, self.y_test)
        self.check_transform(pipe, self.X_constrained_test)
        # we cannot use check_predict because in this case the shape of the
        # output is the shape of X_constrained.X, not X_constrained
        y_predicted = estimator.predict(self.X_constrained)
        self.assertEqual(len(y_predicted), len(self.X_constrained.X))

        # test in a pipeline with PCA
        estimator = clone(self.estimator)
        pipe = make_pipeline(estimator, PCA())
        pipe.fit(self.X_constrained_train, self.y_train)
        self.check_transform(pipe, self.X_constrained_test)

    def test_no_fit_attributes_set_in_init(self):
        """Check that Estimator.__init__ doesn't set trailing-_ attributes."""
        # From scikit-learn
        estimator = clone(self.estimator)
        for attr in dir(estimator):
            if attr.endswith("_") and not attr.startswith("__"):
                # This check is for properties, they can be listed in dir
                # while at the same time have hasattr return False as long
                # as the property getter raises an AttributeError
                assert_false(
                    hasattr(estimator, attr),
                    "By convention, attributes ending with '_' are "
                    'estimated from data in scikit-learn. Consequently they '
                    'should not be initialized in the constructor of an '
                    'estimator but in the fit method. Attribute {!r} '
                    'was found in estimator {}'.format(
                        attr, type(estimator).__name__))

    def test_estimators_fit_returns_self(self):
        """Check if self is returned when calling fit"""
        # From scikit-learn
        estimator = clone(self.estimator)
        assert_true(estimator.fit(self.X_constrained, y) is estimator)

    def test_pipeline_consistency(self):
        # From scikit learn
        # check that make_pipeline(est) gives same score as est
        estimator = clone(self.estimator)
        pipeline = make_pipeline(estimator)
        estimator.fit(self.X_constrained, y)
        pipeline.fit(self.X_constrained, y)

        funcs = ["score", "fit_transform"]

        for func_name in funcs:
            func = getattr(estimator, func_name, None)
            if func is not None:
                func_pipeline = getattr(pipeline, func_name)
                result = func(self.X_constrained, y)
                result_pipe = func_pipeline(self.X_constrained, y)
                assert_allclose_dense_sparse(result, result_pipe)

    def test_dict_unchanged(self):
        # From scikit-learn
        estimator = clone(self.estimator)
        if hasattr(estimator, "n_components"):
            estimator.n_components = 1
        estimator.fit(self.X_constrained, y)
        for method in ["predict", "transform", "decision_function",
                       "predict_proba"]:
            if hasattr(estimator, method):
                dict_before = estimator.__dict__.copy()
                getattr(estimator, method)(self.X_constrained)
                assert_dict_equal(estimator.__dict__, dict_before,
                                  'Estimator changes __dict__ during %s'
                                  % method)

    def test_dont_overwrite_parameters(self):
        # From scikit-learn
        # check that fit method only changes or sets private attributes
        estimator = clone(self.estimator)
        if hasattr(estimator, "n_components"):
            estimator.n_components = 1
        dict_before_fit = estimator.__dict__.copy()

        estimator.fit(self.X_constrained, y)
        dict_after_fit = estimator.__dict__

        public_keys_after_fit = [key for key in dict_after_fit.keys()
                                 if is_public_parameter(key)]

        attrs_added_by_fit = [key for key in public_keys_after_fit
                              if key not in dict_before_fit.keys()]

        # check that fit doesn't add any public attribute
        assert_true(not attrs_added_by_fit,
                    ('Estimator adds public attribute(s) during'
                     ' the fit method.'
                     ' Estimators are only allowed to add private '
                     'attributes'
                     ' either started with _ or ended'
                     ' with _ but %s added' % ', '.join(
                        attrs_added_by_fit)))

        # check that fit doesn't change any public attribute
        attrs_changed_by_fit = [key for key in public_keys_after_fit
                                if (dict_before_fit[key]
                                    is not dict_after_fit[key])]

        assert_true(not attrs_changed_by_fit,
                    ('Estimator changes public attribute(s) during'
                     ' the fit method. Estimators are only allowed'
                     ' to change attributes started'
                     ' or ended with _, but'
                     ' %s changed' % ', '.join(attrs_changed_by_fit)))

class _TestPairsBase(_TestWeaklySupervisedBase):

    def setUp(self):
        self.num_points_in_constraint = 2
        super(_TestPairsBase, self).setUp()


class _TestQuadrupletsBase(_TestWeaklySupervisedBase):

    def setUp(self):
        self.num_points_in_constraint = 4
        super(_TestQuadrupletsBase, self).setUp()


class TestITML(_TestPairsBase):

    def setUp(self):
        self.estimator = ITML()
        super(TestITML, self).setUp()

    def test_sklearn_check_estimator(self):
        check_estimator(SklearnPredictorWrapper())

class TestLSML(_TestQuadrupletsBase):

    def setUp(self):
        self.estimator = LSML()
        super(TestLSML, self).setUp()

class TestMMC(_TestPairsBase):

    def setUp(self):
        self.estimator = MMC()
        super(TestMMC, self).setUp()

class TestSDML(_TestPairsBase):
    
    def setUp(self):
        self.estimator = SDML()
        super(TestSDML, self).setUp()


class SklearnPredictorWrapper(ITML):
    def make_constrained_dataset(self, X):
        # for now it fails if accept_sparse=True, to be fixed
        X = check_array(X, accept_sparse=False)

        # Here is the part where we want to build a ConstrainedDataset from an
        # array-like dataset. Note that in
        # sklearn.utils.estimator_checks.check_methods_subset_invariance, we
        # test that an algorithm gives the same results if applied sample by
        # sample or on all samples at once, so we may want to build a
        # ConstrainedDataset where each pair depends only on one sample

        # this solution: link every point to a point that has twice its
        # coordinates
        X_bis = np.vstack([X, 2 * X])
        c = np.hstack([np.arange(X.shape[0])[:, None],
                       np.arange(X.shape[0])[:, None] + X.shape[0]])
        # other idea: maybe link every point to a point that has a permutation
        #  of its feature values ...

        X_constrained = ConstrainedDataset(X_bis, c)
        return X_constrained

    def fit(self, X, y, random_state=np.random):
        X_constrained = self.make_constrained_dataset(X)
        _, y = check_X_y(X, y)
        # a PairsMetricLearner should learn on a binary y
        y_bis = y.astype(bool)
        return super(SklearnPredictorWrapper, self).fit(X_constrained, y_bis)

    def score(self, X, y):
        X_constrained = self.make_constrained_dataset(X)
        # a PairsMetricLearner should learn on a binary y
        y_bis = y.astype(bool)
        return super(SklearnPredictorWrapper, self).score(X_constrained,
                                                          y_bis)

    def predict(self, X):
        # sometimes in testing, due to inheritance of methods like predict,
        # score etc that call each other, make_constrained_dataset would be
        # called several times on the input if I remove this condition. There
        # should be a better fix.
        if type(X) is not ConstrainedDataset:
            X_constrained = self.make_constrained_dataset(X)
        else:
            X_constrained = X
        return super(SklearnPredictorWrapper, self).predict(X_constrained)

    def transform(self, X=None):
        # when transforming, the X_constrained as input of transform should
        # only contain X, and not more samples, so we will not use
        # make_constrained_dataset because it creates more samples
        fake_constrained_dataset = ConstrainedDataset(X, [[0, 0]])
        return super(SklearnPredictorWrapper, self).transform(
            fake_constrained_dataset)

    def decision_function(self, X):
        if type(X) is not ConstrainedDataset:
            X_constrained = self.make_constrained_dataset(X)
        else:
            X_constrained = X
        return super(SklearnPredictorWrapper, self).decision_function(
            X_constrained)
