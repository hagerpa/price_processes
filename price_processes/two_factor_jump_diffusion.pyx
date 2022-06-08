import numpy as np

from .price_model import PriceModel


class TwoFactorJumpDiffusion(PriceModel):
    def __init__(self,
                 volatility = (0.2, 0.2),
                 mean = 2.5,
                 mean_reversion=(0.25, 0.5),
                 lam=2.0,
                 jump_mean=6.0,
                 jump_std=2.0,
                 correlation=0.5,
                 euler_refinement = 1):
        """
        Simulation of two correlated jump diffusion processes processes, following the SDE

            dX[0]_t = mean_revision[0] * (mean - X[0]_t) dt - volatility[0] * X_t dW[0]_t - (J[0]_t - X[0]_t) dP_t
            dX[1]_t = mean_revision[1] * (mean - X[0]_t) dt - volatility[1] * X_t dW[1]_t - (J[1]_t - X[1]_t) dP_t

        where W^1 and W^2 are Brownian motions correlated by
        :param correlation,
        which simultaneously defines the correlation of the jumps displacements.

        Additional parameters specifying the jump distribution:
        :param lam: Intensity of the Poisson process.
        :param jump_mean: Mean of jumps.
        :param jump_std: Standard deviation of jumps.

        :return: (n_samples, n_steps, 2)-sample matrix
        """
        super().__init__()
        self.volatility = volatility
        self.lam = lam
        self.jump_mean = jump_mean
        self.jump_std = jump_std
        self.mean = mean
        self.mean_reversion = mean_reversion
        self.euler_refinement = euler_refinement
        self.correlation = correlation

    def sample_paths(self, T:float, n_steps:int, n_samples: int, start_law):
        cdef X0 = PriceModel.start((n_samples, 2), start_law)
        n = n_steps * self.euler_refinement
        m = n_samples
        time_delta = T/(n - 1)
        # Sampling all at one place so that fixing a random seed in a top level gives reasonable results.
        P = np.array(np.random.binomial(1, time_delta*self.lam, size=(m,n)), dtype=np.int16)
        W = np.random.randn(m,n,2)
        c_coefficient = (1-self.correlation**2)**0.5
        W[:,:,1] = self.correlation*W[:,:,0] + c_coefficient*W[:,:,1]
        W = np.sqrt(time_delta)*W
        J = np.random.randn(m,n,2)
        J[:,:,1] = self.correlation*J[:,:,0] + c_coefficient*J[:,:,1]
        J = self.jump_std*J + self.jump_mean

        mr1, mr2 = self.mean_reversion
        vol1, vol2 = self.volatility
        X = np.array(__euler__(T, n_steps * self.euler_refinement, n_samples,
                                    self.mean, mr1, mr2, vol1, vol2, W, J, P, X0))[:,::self.euler_refinement]
        return X

cdef __euler__(double T, int n, int m, double mean, double mr1, double mr2, double vol1, double vol2, double[:,:,:] W,
double[:,:,:] J, short[:,:] P, double[:,:] X0):
    cdef double time_delta = T/(n - 1)
    cdef double[:,:,:] X = np.empty((m, n, 2))
    cdef int j
    for j in range(m):
        X[j, 0, 0] = X0[j,0]
        X[j, 0, 1] = X0[j,1]
        for i in range(1, n):
            X[j, i, 0] = X[j, i-1, 0] + mr1*(mean - X[j, i - 1, 0])*time_delta \
                    + vol1*X[j, i - 1, 0]*W[j, i - 1, 0] \
                    + P[j,i] * (J[j,i,0] - X[j, i - 1, 0])

            X[j, i, 1] = X[j, i-1, 1] + mr2*(X[j, i - 1, 0] - X[j, i - 1, 1])*time_delta \
                    + vol2*X[j, i - 1, 1]*W[j, i - 1, 1] \
                    + P[j,i] * (J[j,i,1] - X[j, i - 1, 1])
    return X
