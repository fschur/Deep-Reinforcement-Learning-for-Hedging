import numpy as np
import space_bandits
from financial_models.option_price_models import BSM
from financial_models.asset_price_models import GBM
import random
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

seed = 345
np.random.seed(seed)
torch.manual_seed(seed)


volatility = 0.15
strike_price = 1
starting_price = 1
mu = 0.0
T = 1.0
num_steps = 128
dt = T/num_steps
risk_free_interest_rate = 0.01
trading_cost_para = 0 #1/1000
norm_factor = 10000

apm = GBM(mu=mu, dt=dt, s_0=starting_price, sigma=volatility)
opm = BSM(strike_price=strike_price, risk_free_interest_rate=risk_free_interest_rate, volatility=volatility, T=T, dt=dt)

num_actions = 21
model = space_bandits.NeuralBandits(num_actions, 2, layer_sizes=[20, 20], initial_pulls=20,
                                    lr_decay_rate=0, initial_lr=0.005, reset_lr=False, memory_size=50000,
                                    training_freq=10, training_freq_network=50)

num_eps = 150


def generate_data(apm, opm, num_steps, dt, n=3):
    D = []
    for j in range(n):
        apm.reset()
        price = apm.get_current_price()
        opt_price = opm.compute_option_price(T, price, mode="ttm")
        for p in range(num_steps):
            apm.compute_next_price()
            next_price = apm.get_current_price()
            next_opt_price = opm.compute_option_price(T-(p+1)*dt, next_price, mode="ttm")
            delta = opm.compute_delta_ttm(T-p*dt, price)
            D.append({"p": price, "np": next_price, "op": opt_price, "nop": next_opt_price, "ttm": T - p*dt,
                      "nttm": T - (p+1)*dt, "delta": delta})
            price = next_price
            opt_price = next_opt_price
    return D


def test(model, apm, opm, num_steps, dt):
    D = generate_data(apm, opm, num_steps, dt, n=10)
    rewards = []
    delta_rewards = []
    for tupel in D:
        inp = np.array([[tupel["p"], tupel["ttm"]]], dtype=np.float)
        out = model.predict(inp, thompson=False)[0]
        action = out / num_actions
        reward = -((tupel["nop"] - tupel["op"]) - action * (tupel["np"] - tupel["p"]))**2
        #reward = -(action - tupel["delta"]) ** 2
        rewards.append(reward)

        delta_reward = ((tupel["nop"] - tupel["op"]) - tupel["delta"] * (tupel["np"] - tupel["p"]))**2
        delta_rewards.append(delta_reward)
    return rewards, delta_rewards, np.mean(rewards), np.mean(delta_rewards)


for i in range(num_eps):
    print("episode: ", i)
    if i >= 1:
        losses, delta_losses, test_result, delta_test_results = test(model, apm, opm, num_steps=num_steps, dt=dt)
        print("test_result: ", test_result, delta_test_results)
    D = generate_data(apm, opm, num_steps, dt, n=10)
    random.shuffle(D)

    rewards = []
    for tupel in D:
        inp = np.array([tupel["p"], tupel["ttm"]], dtype=np.float)
        out = model.action(inp)
        action = out / num_actions
        reward = -norm_factor*((tupel["nop"] - tupel["op"]) - action * (tupel["np"] - tupel["p"]))**2
        #reward = -norm_factor*(action - tupel["delta"])**2
        rewards.append(reward)
        model.update(inp, out, reward)
        #loss += norm_factor*torch.pow((tupel["nop"] - tupel["op"]) - out * (tupel["np"] - tupel["p"]), 2)

    print(np.mean(rewards))

for i in range(8):
    apm.reset()
    price = []
    actions = []
    delta = []
    for p in range(num_steps):
        price.append(apm.get_current_price())
        delta.append(opm.compute_delta_ttm(T-p*dt, price[-1]))
        inp = np.array([[delta[-1], price[-1], T - dt*p]], dtype=np.float)
        out = model.predict(inp, thompson=False)[0]
        action = out / num_actions
        actions.append(action)

        apm.compute_next_price()

    plt.plot(price, color="blue")
    plt.plot(delta, color="green")
    plt.plot(actions, color="red")
    plt.savefig("bandit_more_layers" + str(i) + ".png")
    plt.show()