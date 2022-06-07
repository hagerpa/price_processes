from abc import ABC, abstractmethod
from typing import Union, Tuple

import numpy as np


# TODO: Make a uniform understanding for the volatility factor
# TODO: Start laws for d-dim

class PriceModel(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def sample_paths(self, T: float, n_steps: int, n_samples: int, start_law):
        pass

    @staticmethod
    def start(n_samples: Union[int, Tuple[int, int]], start_law):
        if type(n_samples) is int:
            sample_shape = (n_samples, 1)
        else:
            sample_shape = n_samples
        if start_law[0] == "uniform":
            _, a, b = start_law
            X0 = np.random.uniform(a, b, sample_shape)
        elif start_law[0] == "normal":
            _, mean, var = start_law
            X0 = mean + np.random.randn(*sample_shape) * np.sqrt(var)
        elif start_law[0] == "dirac":
            X0 = np.ones(sample_shape) * start_law[1]
        elif start_law[0] == "diracs":
            if type(n_samples) is int:
                if len(start_law[1]) != n_samples:
                    raise AttributeError("The number of dirac laws must equal the number of samples.")
            elif np.shape(start_law[0]) != sample_shape:
                raise AttributeError("The number of dirac laws must have the same number of samples and dimensions as"
                                     "samples.")
            X0 = np.reshape(start_law[1], sample_shape)
        else:
            raise AttributeError("Unknown start distribution " + str(start_law))
        return X0

    def sub_sample_paths(self, Xt, dt, n_sub_samples):
        pass
