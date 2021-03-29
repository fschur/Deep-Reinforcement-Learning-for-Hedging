import numpy as np
from abc import ABC, abstractmethod


class GenericAssetPriceModel(ABC):
    @abstractmethod
    def get_current_price(self):
        pass

    @abstractmethod
    def compute_next_price(self, *action):
        pass

    @abstractmethod
    def reset(self):
        pass


class GBM(GenericAssetPriceModel):
    def __init__(self, mu=0, dt=0.1, s_0=100, sigma=0.2):
        self.mu = mu
        self.dt = dt
        self.s_0 = s_0
        self.sigma = sigma
        self.current_price = s_0

    def compute_next_price(self):
        i = np.random.normal(0, np.sqrt(self.dt))
        new_price = self.current_price * np.exp((self.mu - self.sigma ** 2 / 2) * self.dt
                   + self.sigma * i)
        self.current_price = new_price

    def reset(self):
        self.current_price = self.s_0

    def get_current_price(self):
        return self.current_price