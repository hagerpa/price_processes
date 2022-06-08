import numpy as np

from price_processes.price_model import PriceModel


class OrnsteinUhlenbeck(PriceModel):
    def __init__(self, volatility = 0.5, mean = 0.0, mean_revision = 0.9):
        """
        Simulation of an Ornstein-Uhlenbeck process using a Euler discretization of the SDE
            dX_t = mean_revision * (mean - X_t) dt + volatility dW_t.

        :return: (n_samples, n_steps, 1)-sample matrix
        """
        super().__init__()
        self.volatility = volatility
        self.mean = mean
        self.mean_revision = mean_revision
        self.interest_rate = 0

    def sample_paths(self, T:float, n_steps:int, n_samples: int, start_law):
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
                X[j, i] = self.mean_revision * (X[j, i - 1] - self.mean) + self.volatility * W[j, i - 1]

        return np.expand_dims(PriceModel.start(n_samples, start_law) + X, axis=2)



class LogOrnsteinUhlenbeck(OrnsteinUhlenbeck):
    def __init__(self, volatility = 0.5, mean = 0.0, mean_revision = 0.9):
        """
        Simulation of a exponentiated Ornstein-Uhlenbeck process using a Euler discretization of the SDE
            d log(X_t) = mean_revision * (mean - log(X_t)) dt + volatility dW_t.
        :param volatility:
        :param mean:
        :param mean_revision:
        """
        super().__init__(volatility, mean, mean_revision)

    def sample_paths(self, T:float, n_steps:int, n_samples: int, start_law):
        return PriceModel.start(n_samples, start_law) * \
               np.exp(OrnsteinUhlenbeck.sample_paths(self, T, n_steps, n_samples, ('dirac',0)))
