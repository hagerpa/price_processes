import unittest

import numpy as np
from numpy.testing import assert_allclose


class TestBlackScholes(unittest.TestCase):

    def test_zero_volatility(self):
        from optimal_control.price_models import BlackScholesModel
        T = 1
        n_steps = 11

        # one-dimensional
        model = BlackScholesModel(volatility=0.0, drift=1.0)
        X = model.sample_paths(T, n_steps, n_samples=1, start_law=('dirac', 1))
        assert np.allclose(X[0, :], np.exp(np.linspace(0, T, n_steps)))

        # two-dimensional
        model = BlackScholesModel(volatility=0.0, drift=1.0, correlation=np.eye(2))
        X = model.sample_paths(T, n_steps, n_samples=2, start_law=('dirac', 1))
        assert X.shape == (2, 11, 2)
        assert np.allclose(X[1, :, 0], np.exp(np.linspace(0, T, n_steps)))
        assert np.allclose(X[0, :, 1], np.exp(np.linspace(0, T, n_steps)))

    def test_variance_test(self):
        from optimal_control.price_models import BlackScholesModel
        T = 1
        n_steps = 31
        n_samples = 100_000

        # one-dimensional
        model = BlackScholesModel(volatility=0.2, drift=0.0)
        X = model.sample_paths(T, n_steps, n_samples=n_samples, start_law=('dirac', 1))
        assert_allclose(np.log(X[:, n_steps - 1]).std(), model.volatility, rtol=1 / np.sqrt(n_samples))

        # two-dimensional
        rho = 0.5
        C = (1 - rho) * np.eye(2) + rho * np.ones((2, 2))
        model = BlackScholesModel(volatility=0.2, drift=0.0, correlation=C)
        X = model.sample_paths(T, n_steps, n_samples=n_samples, start_law=('dirac', 1))

        var_ = np.log(X[:, n_steps - 1, 1]).var()
        assert_allclose(var_, model.volatility ** 2, rtol=3 / np.sqrt(n_samples))

        corr_ = np.corrcoef(np.log(X[:, n_steps - 1, 1]), np.log(X[:, n_steps - 1, 0]))
        assert_allclose(corr_, C, rtol=3 / np.sqrt(n_samples))


if __name__ == '__main__':
    unittest.main()
