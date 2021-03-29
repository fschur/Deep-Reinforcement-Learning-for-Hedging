import numpy as np
import torch
import matplotlib.pyplot as plt
from financial_models.asset_price_models import GBM
from financial_models.option_price_models import BSM
import tensorflow as tf
from machin.frame.algorithms import DQN
from torch import nn
import gym
import time
import gym_hedging

seed = 345
np.random.seed(seed)
torch.manual_seed(seed)
tf.random.set_seed(seed)

tim = time.time()

mu = 0
dt = 1/128
T = 1
num_steps = T/dt
s_0 = 1
strike_price = s_0
sigma = 0.15
r = 0.01

apm = GBM(mu=mu, dt=dt, s_0=s_0, sigma=sigma)
opm = BSM(strike_price=strike_price, risk_free_interest_rate=r, volatility=sigma, T=T, dt=dt)
env = gym.make('hedging-v0', asset_price_model=apm, dt=dt, T=T, num_steps=num_steps, trading_cost_para=1,
                     L=1, strike_price=strike_price, int_holdings=False, initial_holding=0, mode="PL",
               option_price_model=opm)

num_actions = 21


class QNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_num):
        super(QNet, self).__init__()

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_num)

    def forward(self, state):
        a = torch.relu(self.fc1(state))
        a = torch.relu(self.fc2(a))
        return self.fc3(a)

qnet = QNet(4, 20, num_actions)
qnet_t = QNet(4, 20, num_actions)

dqn = DQN(qnet, qnet_t,
          torch.optim.Adam,
          nn.MSELoss(reduction='sum'), discount=0.8, epsilon_decay=0.999, learning_rate=0.001,
          lr_scheduler=torch.optim.lr_scheduler.StepLR, lr_scheduler_kwargs=[{"step_size": 1000*128}])

num_eps = 5000
norm_factor = 10000000


def test_delta(n=10):
    rew = []
    for i in range(n):
        state = env.reset()
        done = False
        state = state[[0, 1, 2, 4]]
        while not done:
            action = state[3] - env.h
            new_state, reward, done = env.step(action)
            reward = np.sum(reward)
            new_state = new_state[[0, 1, 2, 4]]
            reward = -(reward) ** 2 + 1 / 1000 * reward
            #reward = -(action + env.h - state[3]) ** 2  # remove that
            rew.append(reward)
            state = new_state
    return np.mean(rew)


def test(n=10):
    rew = []
    for i in range(n):
        state = env.reset()
        done = False
        state = state[[0, 1, 2, 4]]
        while not done:
            out = dqn.act_discrete({"state": torch.tensor(state, dtype=torch.float32).unsqueeze(0)})
            action = out.squeeze().detach().numpy() / num_actions
            new_state, reward, done = env.step(action - env.h)
            reward = np.sum(reward)
            new_state = new_state[[0, 1, 2, 4]]
            reward = -(reward) ** 2 + 1 / 1000 * reward
            #reward = -(action + env.h - state[3]) ** 2  # remove that
            if i == 1:
                print(action, state[3])
            rew.append(reward)
            state = new_state
    return np.mean(rew)

rew = []

for j in range(num_eps):
    print("episode: ", j)
    state = env.reset()
    done = False
    state = state[[0,1,2,4]]
    while not done:
        out = dqn.act_discrete_with_noise({"state": torch.tensor(state, dtype=torch.float32).unsqueeze(0)})
        action = out.squeeze().detach().numpy()/num_actions - env.h
        new_state, reward, done = env.step(action)
        #print(action)
        #print(reward)
        reward = np.sum(reward)
        #print(reward)
        new_state = new_state[[0, 1, 2, 4]]
        #print(state)
        reward = -norm_factor*((reward) ** 2 + 1 / 1000 * reward)
        rew.append(reward)

        dqn.store_transition({
            "state": {"state": torch.tensor(state, dtype=torch.float32).unsqueeze(0)},
            "action": {"action": out},
            "next_state": {"state": torch.tensor(new_state, dtype=torch.float32).unsqueeze(0)},
            "reward": float(reward),  # norm factor
            "terminal": done
        })
        state = new_state

    if j % 50 == 0 and j != 0:
        print(test(10), test_delta(10))
        print("reward: ", np.mean(rew), np.mean(rew)/norm_factor)
        rew = []

    if j > 100:
        for _ in range(int(num_steps)):
            dqn.update()

#dqn.save("dqn_model_1000")

rew_m_l = []
cost_m_l = []
for j in range(100):
    rew = []
    cost_l = []
    state = env.reset()
    done = False
    state = state[[0,1,2,4]]
    while not done:
        out = dqn.act_discrete({"state": torch.tensor(state, dtype=torch.float32).unsqueeze(0)})
        action = out.squeeze().detach().numpy()/num_actions - env.h
        new_state, reward, done = env.step(action)
        cost = reward[1]
        reward = np.sum(reward)

        new_state = new_state[[0, 1, 2, 4]]
        rew.append(reward)
        cost_l.append(cost)
        state = new_state
    rew_m_l.append(np.sum(rew))
    cost_m_l.append(np.sum(cost_l))
print("__________")
print(rew_m_l)
print(np.mean(rew_m_l), np.std(rew_m_l))
print("__________")
print(cost_m_l)
print(np.mean(cost_m_l), np.std(cost_m_l))

rew_m_l = []
cost_m_l = []
for j in range(100):
    rew = []
    cost_l = []
    state = env.reset()
    done = False
    state = state[[0, 1, 2, 4]]
    while not done:
        action = state[3] - env.h
        new_state, reward, done = env.step(action)
        cost = reward[1]
        reward = np.sum(reward)

        new_state = new_state[[0, 1, 2, 4]]
        rew.append(reward)
        cost_l.append(cost)
        state = new_state
    rew_m_l.append(np.sum(rew))
    cost_m_l.append(np.sum(cost_l))
print("__________")
print(rew_m_l)
print(np.mean(rew_m_l), np.std(rew_m_l))
print("__________")
print(cost_m_l)
print(np.mean(cost_m_l), np.std(cost_m_l))
print("__________")
print(tim - time.time())

