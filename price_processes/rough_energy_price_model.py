import numpy as np

from optimal_control.price_models import PriceModel


# TODO: Include Bennedsen Model

class RoughEnergyPriceModel(PriceModel):
    def __init__(self, volatility: float, H: float, lam: float, memory_length: int, include_past_frame: bool):
        super().__init__()
        self.lam = lam
        self.H = H
        self.volatility = volatility
        self.memory_length = memory_length
        self.dimension = 1 if (include_past_frame is False) else memory_length

    def sample_paths(self, T: float, n_steps: int, n_samples: int, start_law):
        time_delta = T / (n_steps - 1)
        r = time_delta * self.memory_length

        noise = np.random.randn(n_samples, n_steps - 1 + self.memory_length) * time_delta ** 0.5
        kernel = np.linspace(0, r, self.memory_length)[1:]
        kernel = (kernel ** (self.H - 0.5)) * np.exp(-self.lam * kernel)

        X = (self.volatility ** 0.5) * np.apply_along_axis(lambda v: np.convolve(v, kernel, 'valid'), axis=1, arr=noise)
        return X
