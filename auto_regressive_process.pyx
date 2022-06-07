import numpy as np

from .price_model import PriceModel


class LogAutoRegressiveProcess(PriceModel):
    def __init__(self, volatility = 0.5, mu = 0.0, k = 0.9, interest_rate = 0):
        super().__init__()
        self.volatility = volatility
        self.mu = mu
        self.k = k
        self.interest_rate = 0

    def sample_paths(self, T:float, n_steps:int, n_samples: int, start_law):
        S0 = PriceModel.start(n_samples, start_law)

        cdef int n = n_steps
        cdef int m = n_samples

        cdef double time_delta = T/(n - 1)
        cdef double[:,::1] X = np.empty((m, n))
        cdef int j

        # Sampling all at one place so that fixing a random seed in a top level gives reasonable results.
        cdef double[:,::1] W = np.sqrt(time_delta)*np.random.randn(m, n)

        for j in range(m):
            X[j, 0] = 0
            for i in range(1, n):
                X[j, i] = (1-self.k)*(X[j, i-1] - self.mu) + self.mu + self.volatility*W[j, i - 1]

        return np.exp(X)*S0 * np.exp(-np.linspace(0,T,n_steps) * self.interest_rate) #discounting
