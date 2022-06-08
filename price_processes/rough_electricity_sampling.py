import numpy as np
from optimal_control.price_models import PriceModel
from scipy.fft import rfft, hfft, ihfft
from scipy.special import gamma, kv


class BennedsenElectricityModel(PriceModel):
    def __init__(self, standard_deviation: float, mean: float, alpha: float, mean_reversion: float):
        """
        Simulation of the rough electricity Price model proposed in [M. Bennedsen. A rough multi-factor model of
        electricity spot prices. Energy Economics, 63:301–313, March 2017.], i.e., of the mean reverting fractional
        processes given by the volterra integral

        X_t = mean + standard_deviation/constant int_{-infinity}^t (t - s)**alpha * exp(-mean_reversion * (t-s)) dW_s,

        where W is a Brownian motion.

        :return: (n_samples, n_steps, 1)-sample matrix
        """
        super().__init__()
        self.mean_reversion = mean_reversion
        self.alpha = alpha
        self.standard_deviation = standard_deviation
        self.mean = mean
        self.dimension = 1

    def auto_correlation(self, h):
        a = self.alpha
        k = self.mean_reversion
        const = 2 ** (- a + 0.5) * k ** (a + 0.5) / gamma(a + 0.5)
        return const * h ** (a + 0.5) * np.exp(- k * h) * kv(a + 0.5, k * h)

    def sample_paths(self, T: float, n_steps: int, n_samples: int, start_law):
        c = self.auto_correlation(np.linspace(0, T, n_steps))
        gam = np.concatenate([c, c[1:-1][::-1]])
        lam = np.sqrt(rfft(gam))

        Z = np.random.randn(n_samples, len(gam))

        if start_law == 'stationary':
            X_0 = self.mean
        else:
            X_0 = self.start(n_samples, start_law)
            Z[:, 0] = (X_0 - self.mean) / self.standard_deviation

        return X_0 + self.standard_deviation * hfft(lam * ihfft(Z))[:, :n_steps]
