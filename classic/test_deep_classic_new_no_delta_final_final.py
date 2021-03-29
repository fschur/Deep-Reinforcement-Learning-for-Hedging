import numpy as np
import torch.nn as nn
import torch
import random
import torch.nn.functional as F
from financial_models.option_price_models import BSM
from financial_models.asset_price_models import GBM
import matplotlib.pyplot as plt


class FullyConnected(nn.Module):
    def __init__(self, input, hidden, out_size, num_layers, f):
        super(FullyConnected, self).__init__()

        self.num_layers = num_layers

        self.first_layer = nn.Linear(input, hidden)

        self.linear = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(num_layers)])

        self.f = f

        self.out_layer = nn.Linear(hidden, out_size)

    def forward(self, x):
        x = self.f(self.first_layer(x))
        for layer in range(self.num_layers):
            x = self.linear[layer](x)
            x = self.f(x)

        x = self.f(x)
        x = self.out_layer(x)

        return x


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
    losses = []
    delta_losses = []
    for tupel in D:
        inp = torch.tensor(np.array([tupel["p"], tupel["ttm"]])).double()
        out = model(inp)
        loss = torch.pow((tupel["nop"] - tupel["op"]) - out * (tupel["np"] - tupel["p"]), 2)
        losses.append(loss.detach().numpy())

        delta_loss = ((tupel["nop"] - tupel["op"]) - tupel["delta"] * (tupel["np"] - tupel["p"]))**2
        delta_losses.append(delta_loss)
    return losses, delta_losses, np.mean(losses), np.mean(delta_losses)




volatility = 0.15
strike_price = 1
starting_price = 1
mu = 0.0
T = 1.0
num_steps = 128
dt = T/num_steps
risk_free_interest_rate = 0.01

seed = 345
np.random.seed(seed)
torch.manual_seed(seed)

model = FullyConnected(2, 16, 1, 5, f=torch.nn.functional.relu)
model.double()
apm = GBM(mu=mu, dt=dt, s_0=starting_price, sigma=volatility)
opm = BSM(strike_price=strike_price, risk_free_interest_rate=risk_free_interest_rate, volatility=volatility, T=T, dt=dt)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

num_eps = 1200
norm_factor = 10000

test_res = []
test_res_delta = []

for i in range(num_eps):
    print("episode: ", i)

    if i % 20 == 0:
        losses, delta_losses, test_result, delta_test_results = test(model, apm, opm, num_steps=num_steps, dt=dt, n=20)
        print("test_result: ", test_result, delta_test_results)
        test_res.append(test_result)
        test_res_delta.append(delta_test_results)
    D = generate_data(apm, opm, num_steps, dt, n=10)
    random.shuffle(D)

    if i == 300:
        for q in optimizer.param_groups:
            q["lr"] = 0.001

    if i == 600:
        for q in optimizer.param_groups:
            q["lr"] = 0.0001

    if i == 900:
        for q in optimizer.param_groups:
            q["lr"] = 0.00001

    loss = 0
    for tupel in D:
        inp = torch.tensor(np.array([tupel["p"], tupel["ttm"]]), dtype=torch.float64)
        out = model(inp)
        loss += norm_factor*torch.pow((tupel["nop"] - tupel["op"]) - out * (tupel["np"] - tupel["p"]), 2)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print("loss:", loss.detach().numpy())

print(test_res)
print(test_res_delta)
plt.plot(test_res, label="NN")
plt.plot(test_res_delta, label="delta")
#plt.xlabel("")
plt.xlabel("loss")
plt.legend()
plt.savefig("test_deep_classic_new_no_delta_losses_final.png")

torch.save(model, "model_classic_final.pth")

for i in range(5):
    out = []
    delta = []
    plt.figure()
    D = generate_data(apm, opm, num_steps, dt, n=1)
    for tupel in D:
        inp = torch.tensor(np.array([tupel["p"], tupel["ttm"]])).double()
        out.append(model(inp).detach().numpy())
        delta.append(tupel["delta"])
    plt.plot([i/len(out) for i in range(len(out))], out, label="NN", color="green")
    plt.plot([i/len(delta) for i in range(len(delta))], delta, label="delta", color="red")
    plt.xlabel("time")
    #plt.xlabel("")
    plt.savefig("test_deep_classic_new_no_delta_final" + str(i) + ".png")
    plt.show()
    plt.clf()
    plt.cla()
    plt.close()

test_res = []
for i in range(20):
    losses, delta_losses, test_result, delta_test_results = test(model, apm, opm, num_steps=num_steps, dt=dt, n=1)
    test_res.append(test_result)

print(test_res)
print(np.mean(test_res), np.std(test_res))


def test_2(model, apm, opm, num_steps, dt, n=10):
    D = generate_data(apm, opm, num_steps, dt, n=1)
    losses = []
    delta_losses = []
    for tupel in D:
        inp = torch.tensor(np.array([tupel["p"], tupel["ttm"]])).double()
        out = model(inp)
        loss = (-tupel["nop"] + tupel["op"]) + out * (tupel["np"] - tupel["p"])
        losses.append(loss.detach().numpy())

        delta_loss = ((tupel["nop"] - tupel["op"]) - tupel["delta"] * (tupel["np"] - tupel["p"]))**2
        delta_losses.append(delta_loss)
    return losses, delta_losses, np.mean(losses), np.mean(delta_losses)

test_res = []
for i in range(20):
    losses, delta_losses, test_result, delta_test_results = test_2(model, apm, opm, num_steps=num_steps, dt=dt, n=1)
    test_res.append(test_result)

print(test_res)
print(np.mean(test_res), np.std(test_res))
