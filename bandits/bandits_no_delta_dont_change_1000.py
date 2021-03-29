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
norm_factor = 10000000

apm = GBM(mu=mu, dt=dt, s_0=starting_price, sigma=volatility)
opm = BSM(strike_price=strike_price, risk_free_interest_rate=risk_free_interest_rate, volatility=volatility, T=T, dt=dt)

num_actions = 21
model = space_bandits.NeuralBandits(num_actions, 2, layer_sizes=[20, 20], initial_pulls=20, lr_decay_rate=1,
                                    reset_lr=True)

num_eps = 60


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


def test(model, apm, opm, num_steps, dt, n=10):
    D = generate_data(apm, opm, num_steps, dt, n=n)
    rewards = []
    delta_rewards = []
    for tupel in D:
        inp = np.array([[tupel["p"], tupel["ttm"]]], dtype=np.float)
        out = model.predict(inp, thompson=False)[0]
        action = out / num_actions
        pl = (-tupel["nop"] + tupel["op"]) + action * (tupel["np"] - tupel["p"])
        reward = -(pl)**2 + 1/1000*pl
        rewards.append(reward)

        pl_delta = (-tupel["nop"] + tupel["op"]) + tupel["delta"] * (tupel["np"] - tupel["p"])
        delta_reward = -(pl_delta)**2 + 1/1000*pl_delta
        delta_rewards.append(delta_reward)
    return rewards, delta_rewards, np.mean(rewards), np.mean(delta_rewards)


for i in range(num_eps):
    print("episode: ", i)
    if i >= 1:
        losses, delta_losses, test_result, delta_test_results = test(model, apm, opm, num_steps=num_steps, dt=dt, n=10)
        print("test_result: ", test_result, delta_test_results)
    D = generate_data(apm, opm, num_steps, dt, n=10)
    random.shuffle(D)

    rewards = []
    for tupel in D:
        inp = np.array([tupel["p"], tupel["ttm"]], dtype=np.float)
        out = model.action(inp)
        action = out / num_actions
        pl = (-tupel["nop"] + tupel["op"]) + action * (tupel["np"] - tupel["p"])
        reward = norm_factor*(-(pl)**2 + 1/1000*pl)
        rewards.append(reward)
        model.update(inp, out, reward)

    print(np.mean(rewards))

for i in range(8):
    apm.reset()
    plt.figure()
    price = []
    actions = []
    delta = []
    for p in range(num_steps):
        price.append(apm.get_current_price())
        delta.append(opm.compute_delta_ttm(T-p*dt, price[-1]))
        inp = np.array([[price[-1], T - dt*p]], dtype=np.float)
        out = model.predict(inp, thompson=False)[0]
        action = out / num_actions
        actions.append(action)

        apm.compute_next_price()

    plt.plot(price, color="blue")
    plt.plot(delta, color="green")
    plt.plot(actions, color="red")
    plt.savefig("bandit_1000" + str(i) + ".png")
    plt.clf()
    plt.cla()
    plt.close()

fh = []
for i in range(20):
    losses, delta_losses, test_result, delta_test_results = test(model, apm, opm, num_steps=num_steps, dt=dt, n=1)
    fh.append(test_result)
    print(test_result)

print(np.mean(fh), np.var(fh))

for i in range(8):
    apm.reset()
    price = []
    actions = []
    delta = []
    for p in range(num_steps):
        price.append(apm.get_current_price())
        delta.append(opm.compute_delta_ttm(T-p*dt, price[-1]))
        inp = np.array([[price[-1], T - dt*p]], dtype=np.float)
        out = model.predict(inp, thompson=False)[0]
        action = out / num_actions
        actions.append(action)

        apm.compute_next_price()

    #plt.plot([i/len(out) for i in range(len(out))], price, color="blue")
    plt.plot([i/len(delta) for i in range(len(delta))], delta, color="green")
    plt.plot([i/len(actions) for i in range(len(actions))], actions, color="red")
    plt.savefig("bandit_no_1000" + str(i) + ".png")
    plt.clf()
    plt.cla()
    plt.close()


def test_2(model, apm, opm, num_steps, dt, n):
    D = generate_data(apm, opm, num_steps, dt, n=n)
    rewards = []
    delta_rewards = []
    for tupel in D:
        inp = np.array([[tupel["p"], tupel["ttm"]]], dtype=np.float)
        out = model.predict(inp, thompson=False)[0]
        action = out / num_actions
        reward = (-tupel["nop"] + tupel["op"]) + action * (tupel["np"] - tupel["p"])
        rewards.append(reward)

        delta_reward = ((-tupel["nop"] + tupel["op"]) + tupel["delta"] * (tupel["np"] - tupel["p"]))
        delta_rewards.append(delta_reward)
    return rewards, delta_rewards, np.sum(rewards), np.sum(delta_rewards)

fh = []
fh_del = []
for i in range(20):
    losses, delta_losses, test_result, delta_test_results = test_2(model, apm, opm, num_steps=num_steps, dt=dt, n=1)
    fh.append(test_result)
    fh_del.append(delta_test_results)

print(fh)
print(np.mean(fh), np.std(fh))
print(fh_del)
print(np.mean(fh_del), np.std(fh_del))

model.save("bnn_no_model_1000")
