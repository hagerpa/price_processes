import numpy as np
import scipy
from scipy.fft import hfft, ihfft

from .price_model import PriceModel


class FractionalBrownianMotion(PriceModel):
    def __init__(self, H: float, past: int):
        """
        Simulates a fractional Brownian motion using the spectral embedding method and the fast fourier transform.
        :param H: Hurst parameter.
        :param past: extend dimension of the process by including this number of past values.

        :return: (n_samples, n_steps, 1 + past)-sample matrix
        """
        super().__init__()
        self.past = past
        self.H = H
        self.dimension = past + 1

    def sample_paths(self, T: float, n_steps: int, n_samples, start_law):
        X = np.zeros((n_samples, n_steps, self.dimension))
        B = X[:, :, 0]

        k = np.arange(n_steps - 1)
        cov = 0.5 * (np.abs(k + 1) ** (2 * self.H) + np.abs(k - 1) ** (2 * H)) - np.abs(k) ** (2 * self.H)

        delta = T / (n_steps - 1)

        B[:, 1:] = stationary_gaussian(n_samples, cov)
        np.cumsum(delta ** self.H * B, axis=1, out=B)

        X[:, :, 0] = PriceModel.start(n_samples, start_law) + B

        for i in range(1, self.past + 1):
            X[:, i:, i] = X[:, :-i, 0]

        return X


def stationary_gaussian(m, c):
    gam = np.concatenate([c, c[1:-1][::-1]])
    lam = np.sqrt(scipy.fft.rfft(gam))
    return hfft(lam * ihfft(np.random.randn(m, len(gam))))[:, :n]
