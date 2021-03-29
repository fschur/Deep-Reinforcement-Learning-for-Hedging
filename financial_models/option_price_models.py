import numpy as np
from abc import ABC, abstractmethod
from scipy.stats import norm
import warnings


class GenericOptionPriceModel(ABC):
    """
    Generic option price class. The use this with the gym-hedging environment, the option class needs to have
    a function called 'compute_option_price' that computes the option price for a given the asset price and

    """
    @abstractmethod
    def compute_option_price(self, *inputs):
        pass


class BSM(GenericOptionPriceModel):
    def __init__(self, strike_price, risk_free_interest_rate, volatility, T, dt):
        self.strike_price = strike_price
        self.risk_free_interest_rate = risk_free_interest_rate
        self.volatility = volatility
        self.T = T
        self.dt = dt

    def compute_option_price(self, n, asset_price, mode="step"):
        if mode == "step":
            time_to_maturity = self.T - n * self.dt
        elif mode == "ttm":
            time_to_maturity = n
        else:
            raise ValueError("'mode' must be either 'step' or 'ttm'")

        if time_to_maturity < 1e-7:
            if time_to_maturity != 0.0:
                warnings.warn("'time_to_maturity' is smaller than 1e-7. This can cause numerical instability.")
            return max(0, asset_price - self.strike_price)
        d_1 = (np.log(asset_price / self.strike_price) + (self.risk_free_interest_rate + self.volatility**2 / 2)
               * time_to_maturity) / (self.volatility * np.sqrt(time_to_maturity))
        d_2 = d_1 - self.volatility * np.sqrt(time_to_maturity)
        PVK = self.strike_price * np.exp(-self.risk_free_interest_rate * time_to_maturity)
        option_price = norm.cdf(d_1) * asset_price - norm.cdf(d_2) * PVK
        return option_price

    def compute_delta_ttm(self, ttm, asset_price):
        time_to_maturity = ttm
        d_1 = (np.log(asset_price / self.strike_price) + (self.risk_free_interest_rate + self.volatility**2 / 2)
               * time_to_maturity) / (self.volatility * np.sqrt(time_to_maturity))
        delta = norm.cdf(d_1)
        return delta

    def compute_delta(self, n, asset_price):
        time_to_maturity = self.T - n * self.dt
        d_1 = (np.log(asset_price / self.strike_price) + (self.risk_free_interest_rate + self.volatility**2 / 2)
               * time_to_maturity) / (self.volatility * np.sqrt(time_to_maturity))
        delta = norm.cdf(d_1)
        return delta
