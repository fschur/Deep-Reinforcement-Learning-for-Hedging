import gym
import numpy as np
from gym import error, spaces, utils
from gym.utils import seeding
from gym import spaces


class HedgingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, asset_price_model, dt, T, num_steps=100, trading_cost_para=0.01,
                 L=100, strike_price=None, int_holdings=False, initial_holding=0.5, mode="CF", **kwargs):
        self.action_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float) #!! low -L etc
        if int_holdings:
            self.observation_space = spaces.Tuple((spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.int16), #!!
                                                  spaces.Box(low=0, high=np.inf, shape=(1,)),
                                                  spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.int16)))
        else:
            self.observation_space = spaces.Tuple((spaces.Box(low=0, high=np.inf, shape=(1,)),  # !!
                                                   spaces.Box(low=0, high=np.inf, shape=(1,)),
                                                   spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.int16)))
        assert (mode in ["PL", "CF"]), "Only 'PL' and 'CL' are allowed values for mode."
        self.asset_price_model = asset_price_model
        self.current_price = asset_price_model.get_current_price()
        self.n = 0
        self.T = T
        self.dt = dt
        if mode == "PL":
            self.option_price_model = kwargs.get('option_price_model', None)
            assert (self.option_price_model is not None), "If 'PL' is chosen a option_price_model " \
                                                          "needs be be provided."
            self.current_option_price = self.option_price_model.compute_option_price(self.n, self.current_price)
            self.delta = self.option_price_model.compute_delta(0, self.current_price)
        self.done = False
        self.num_steps = num_steps
        self.mode = mode
        self.L = L
        self.initial_holding = initial_holding
        self.h = initial_holding
        self.int_holdings = int_holdings
        if int_holdings:
            self.h = round(self.h)
        if strike_price:
            self.strike_price = strike_price
        else:
            self.strike_price = self.current_price
        self.trading_cost_para = trading_cost_para

    def _compute_cf_reward(self, new_h, next_price, delta_h):
        reward = -self.current_price * delta_h - \
                 self.trading_cost_para * \
                 np.abs(self.current_price * delta_h)
        if self.done:
            asset_value = next_price * new_h - self.trading_cost_para * \
                          np.abs(next_price * new_h)
            payoff = self.L * max(0, next_price - self.strike_price)
            reward += asset_value - payoff
        return reward

    def _compute_pl_reward(self, new_h, next_price, delta_h):
        new_option_price = self.option_price_model.compute_option_price(self.n, next_price)
        reward_option_price = self.L * ((new_option_price - self.current_option_price) - new_h *
                                        (next_price - self.current_price))
        reward_trading_cost = - self.trading_cost_para * self.dt * (abs(delta_h) + 0.01 * delta_h**2)
        self.current_option_price = new_option_price
        if self.done:
            pass
            #reward_trading_cost += - self.trading_cost_para * abs(new_h * next_price)
        return [reward_option_price, reward_trading_cost]

    def step(self, delta_h):
        if self.int_holdings:
            delta_h = round(delta_h)
        new_h = self.h + delta_h
        self.asset_price_model.compute_next_price()
        next_price = self.asset_price_model.get_current_price()
        self.n += 1
        if self.n == self.num_steps:
            self.done = True
        if self.mode == "CF":
            reward = self._compute_cf_reward(new_h, next_price, delta_h)
        elif self.mode == "PL":
            reward = self._compute_pl_reward(new_h, next_price, delta_h)
            self.delta = self.option_price_model.compute_delta(self.n, self.current_price)
        else:
            assert "error 1"
        self.current_price = next_price
        self.h = new_h
        state = self.get_state()
        return [state, reward, self.done]

    def get_state(self):
        time_to_maturity = self.T - self.n * self.dt
        if self.mode == "PL":
            return np.array([self.h, self.current_price, time_to_maturity, self.current_option_price, self.delta],
                            dtype=float)
        else:
            return np.array([self.h, self.current_price, time_to_maturity], dtype=float)

    def reset(self):
        self.asset_price_model.reset()
        self.n = 0
        self.done = False
        self.current_price = self.asset_price_model.get_current_price()
        self.h = self.initial_holding
        if self.mode == "PL":
            self.current_option_price = self.option_price_model.compute_option_price(self.n, self.current_price)
            self.delta = self.option_price_model.compute_delta(self.n, self.current_price)
        if self.int_holdings:
            self.h = round(self.h)
        state = self.get_state()
        return state

    def render(self, mode='human', close=False):
        return self.get_state()
