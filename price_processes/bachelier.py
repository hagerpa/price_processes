import numpy as np
from optimal_control.price_models import BlackScholesModel


class BachelierModel(BlackScholesModel):
    def __init__(self, volatility=0.2, drift=None, correlation=None):
        """
        A simple implementation of a random walk/Bachelier stock price model with random starting point, i.e. of the
        process volatility * B_t + drift * t, where B is a d-dimensional Brownian motion with
        :param correlation: dxd correlation matrix.

        :param volatility: volatility, possibly a vector.
        :param drift: possibly a vector.

        :return: (n_samples, n_steps, d)-sample matrix
        """
        super().__init__(volatility, drift + volatility ** 2 / 2, correlation)

    def sample_paths(self, T: float, n_steps: int, n_samples: int, start_law):
        X = BachelierModel.sample_paths(self, T, n_steps, n_samples, ('dirac', 1))
        return self.start(n_samples, start_law) + np.log(X)
