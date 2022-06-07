import numpy as np

from .price_model import PriceModel


class BachelierModel(PriceModel):
    # TODO: This could be adapted to d-dim with correlation
    def __init__(self, volatility=0.2, drift=None):
        """
        A simple implementation of a random walk/Bachelier stock price model with random starting point, i.e. of the
        process volatility * B_t + drift * t, where B is a Brownian motion.

        :return: (n_samples, n_steps)-sample matrix
        """
        super().__init__()
        self.volatility = volatility
        self.drift = drift

    def sample_paths(self, T: float, n_steps: int, n_samples: int, start_law):
        X0 = PriceModel.start(n_samples, start_law)

        d_vol = self.volatility * (T / (n_steps - 1)) ** 0.5
        drift = np.linspace(0, T, n_steps) * self.drift
        return np.hstack([X0, X0 + d_vol * np.cumsum(np.random.randn(n_samples, n_steps - 1), axis=1)]) + drift
