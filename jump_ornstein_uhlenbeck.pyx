import numpy as np

from .price_model import PriceModel


class JumpOrnsteinUhlenbeck(PriceModel):
    def __init__(self,
                 volatility = 0.2,
                 mean = 2.5,
                 mean_reversion=0.25,
                 lam=2.0,
                 jump_mean=6.4,
                 jump_std=2,
                 euler_refinement = 1):
        """
        A simple implementation of a Jump diffusion stock price model with random starting point. The jump continuous
        part is driven by a Brownian motion and the jump part is driven by a normally compounded Poisson process. The
        price dynamics are as follows

            dX_t = mean_revision * (mean - X_t) dt - volatility * X_t dW_t - (J_t - X_t) dP_t

        Additional parameters specifying the jump distribution:
        :param lam: Intensity of the Poisson process.
        :param jump_mean: Mean of jumps.
        :param jump_std: Standard deviation of jumps.

        :param euler_refinement: discretizaiton of the euler scheme will be on a grid this times finer then the result.

        :return: (n_samples, n_steps)-sample matrix
        """
        super().__init__()
        self.volatility = volatility
        self.lam = lam
        self.jump_mean = jump_mean
        self.jump_std = jump_std
        self.mean = mean
        self.mean_reversion = mean_reversion
        self.euler_refinement = euler_refinement

    def sample_paths(self, T:float, n_steps:int, n_samples: int, start_law):
        X0 = PriceModel.start(n_samples, start_law)

        cdef int n = n_steps * self.euler_refinement
        cdef int m = n_samples
        cdef double time_delta = T/(n - 1)
        cdef double[:,::1] X = np.empty((m, n))
        cdef int j
        # Sampling all at one place so that fixing a random seed in a top level gives reasonable results.
        cdef long[:,::1] P = np.random.binomial(1, time_delta*self.lam, size=(m,n))
        cdef double[:,::1] W = np.sqrt(time_delta)*np.random.randn(m,n)
        cdef double[:,::1] J = self.jump_std * np.random.randn(m,n) + self.jump_mean

        for j in range(m):
            X[j, 0] = X0[j]
            for i in range(1, n):
                X[j, i] = X[j, i-1] + self.mean_reversion*(self.mean - X[j, i - 1])*time_delta \
                        + self.volatility*X[j, i - 1]*W[j, i - 1] \
                        + P[j,i] * (J[j,i] - X[j, i - 1])


        return np.array(X[:,::self.euler_refinement])
