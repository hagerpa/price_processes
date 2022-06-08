import numpy as np

from .price_model import PriceModel


class BlackScholesModel(PriceModel):
    def __init__(self, volatility=0.2, drift=0.0, correlation=None):
        """
        A simple implementation of a Black-Scholes (geometric BM) stock price model with random starting point, i.e. of
        the process
            X[i]_t = X[i]_0 * exp( volatility W[i]_t - (drift - volatility**2/2)t ),  i = 1,...,d,
        where W is a d-dimensional Brownian motion with
        :param correlation: dxd correlation matrix.

        :param volatility: possibly a vector.
        :param drift: possibly a vector.

        :return: (n_samples, n_steps, d)-sample matrix
        """
        super().__init__()
        self.volatility = volatility
        self.correlation = correlation
        self.drift = drift
        self.dimension = 1 if (correlation is None) else correlation.shape[0]

    def sample_paths(self, T: float, n_steps: int, n_samples, start_law):
        time_space = np.linspace(0, T, n_steps)
        time_delta = T / (n_steps - 1)

        if self.dimension == 1:
            X0 = PriceModel.start(n_samples, start_law)

            noise = np.random.randn(n_samples, n_steps - 1) * time_delta ** 0.5
            brownian_motion = np.hstack([np.zeros((n_samples, 1)), np.cumsum(noise, axis=1)])

            return X0 * np.exp(self.volatility * brownian_motion
                               + (- 0.5 * self.volatility ** 2 + self.drift) * time_space)
        else:
            d = self.dimension
            vol = np.ones(d) * self.volatility if np.ndim(self.volatility) == 0 else self.volatility

            X = np.empty((n_samples, n_steps, d))
            X[:, 0, :] = np.zeros((n_samples, d))

            # Filling with noise
            X[:, 1:, :] = np.random.multivariate_normal(np.zeros(d), self.correlation,
                                                        size=(n_samples, n_steps - 1)) * time_delta ** 0.5

            drift_term = np.reshape((- 0.5 * vol ** 2 + self.drift), (1, d)) * time_space.reshape((n_steps, 1))

            np.cumsum(X, axis=1, out=X)  # Cumulating the noise
            np.multiply(X, vol, out=X)  # Multiplying with volatility
            np.add(X, drift_term, out=X)  # Adding the drift
            np.exp(X, out=X)  # Exponentiation
            np.multiply(X, PriceModel.start((n_samples, d), start_law).reshape(n_samples, 1, d),
                        out=X)  # Multiplying with X0
            return X

    def sub_sample_paths(self, Xt, dt, n_sub_samples):
        m, d = Xt.shape
        X = self.sample_paths(dt, 2, n_sub_samples * m, start_law=("dirac", 1))
        X = X[:, 1:].reshape((n_sub_samples, m, d))
        np.multiply(X, Xt, out=X)
        return X
