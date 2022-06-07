from .price_model import PriceModel
import numpy as np
from numpy import logical_and

# TODO: Check implementation

class JumpDiffusion(PriceModel):
    def __init__(self, volatility=0.2,
                 lam = 1.0, jump_mean = 1.0, jump_std = 1.0,
                 interest_rate=.01, additional_drift=0.0):
        """
        A simple implementation of a Jump diffusion stock price model with random starting point. The jump continuous
        part is driven by a Brownian motion and the jump part is driven by a normally compounded Poisson process. The
        price dynamics are as follows
            dX_t = b*dt -r*X_t dt - sigma*X_t dW_t - (J_t - X_t) dP_t

        :param volatility: Volatility per annum.
        :param lam: Intensity of the Poisson process.
        :param jump_mean: Mean of jumps.
        :param jump_std: Standard deviation of jumps.
        :param interest_rate: Continuously compounded per annum.
        :param additional_drift:

        :return: (n_samples, n_steps)-sample matrix
        """
        super().__init__()
        self.volatility = volatility
        self.interest_rate = interest_rate
        self.lam = lam
        self.jump_mean = 1
        self.jump_std = 1
        self.drift = 0.5 * self.volatility ** 2 + self.interest_rate
        self.additional_drift = additional_drift

    def sample_paths(self, T:float, n_steps:int, n_samples: int, start_law):
        X0 = PriceModel.start(n_samples, start_law)

        cdef int n = n_steps
        cdef int m = n_samples
        cdef double time_delta = T/(n_steps - 1)
        cdef double[:,::1] Y = np.empty((n_samples, n_steps))
        cdef int j
        cdef int n_jumps
        for j in range(m):
            noise = np.random.randn(n_steps - 1) * time_delta**0.5
            Y[j, 0] = np.log(X0[j])
            points = np.random.uniform(0.0, T, np.random.poisson(self.lam))
            for i in range(1, n):
                n_points = np.sum(logical_and(points > i*time_delta, points <= i*time_delta))
                Y[j, i] = Y[j, i - 1]+ noise[i - 1] - self.drift * time_delta
                if n_points > 0:
                    Y[j, i] += points*(self.jump_std * np.random.randn(n_points) + self.jump_mean)

        return np.exp(Y) - self.additional_drift * np.linspace(0.0, T, n_steps)
