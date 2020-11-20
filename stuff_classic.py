import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from option_price_model import BSM
from asset_price_model import GBM
import matplotlib.pyplot as plt


class Highway(nn.Module):
    def __init__(self, input, hidden, out_size, num_layers, f):

        super(Highway, self).__init__()

        self.num_layers = num_layers

        self.first_layer = nn.Linear(input, hidden)

        self.nonlinear = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(num_layers)])

        self.linear = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(num_layers)])

        self.gate = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(num_layers)])

        self.f = f

        self.out_layer = nn.Linear(hidden, out_size)

    def forward(self, x):
        x = self.f(self.first_layer(x))
        for layer in range(self.num_layers):
            gate = F.sigmoid(self.gate[layer](x))

            nonlinear = self.f(self.nonlinear[layer](x))
            linear = self.linear[layer](x)

            x = gate * nonlinear + (1 - gate) * linear

        x = self.f(x)
        x = self.out_layer(x)

        return x


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


volatility = 0.15
strike_price = 1
starting_price = 1
mu = 0.01
num_steps = 50
dt = 5 / (50)
risk_free_interest_rate = 0.0
T = 5.0
trading_cost_para = 1/300

seed = 345
np.random.seed(seed)
torch.manual_seed(seed)

model = Highway(3, 20, 1, 6, f=torch.nn.functional.relu)
apm = GBM(mu=mu, dt=dt, s_0=starting_price, sigma=volatility)
opm = BSM(strike_price=strike_price, risk_free_interest_rate=risk_free_interest_rate, volatility=volatility, T=T, dt=dt)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_eps = 1000

lam = dt

f = []

for i in range(num_eps):
    apm.reset()
    h = 0.5
    price = apm.get_current_price()
    last_price = price
    option_price = opm.compute_option_price(0, price)
    last_option_price = option_price
    o = []
    s = []
    g = []
    g_2 = []

    if i > 500:
        for q in optimizer.param_groups:
            q["lr"] = 0.0001

    if i > 800:
        for q in optimizer.param_groups:
            q["lr"] = 0.00001

    for j in range(0, int(num_steps)):
        apm.compute_next_price()
        price = apm.get_current_price()
        option_price = opm.compute_option_price(j+1, price)
        delta = opm.compute_delta((num_steps - j)*dt, last_price)

        data = torch.tensor(np.array([h, last_price, T - j * dt])).float()
        output = model(data)

        loss = torch.pow((option_price - last_option_price + (output + h) * (price - last_price)), 2) + trading_cost_para * lam * (torch.abs(output) + 0.01 * torch.pow(output, 2)) #!!
        #print(loss.detach().numpy().item(), lam*(abs(output.detach().numpy().item()) + 0.01 * output.detach().numpy().item()**2) / 10)
        t_2 = (option_price - last_option_price + delta * (price - last_price))**2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        s.append(delta)
        o.append(output.detach().numpy().item() + h) #!!
        g.append(t_2)
        g_2.append(loss.detach().numpy().item())

        #print("***")
        #print(loss.detach().numpy().item(), t_2)
        #print(output.detach().numpy().item(), delta - h)
        #print(h)

        last_price = price
        last_option_price = option_price
        h += output.detach().numpy().item() #!!
    print(i)
    #print(s)
    #print(o)
    print(np.mean(np.array(g)), np.mean(np.array(g_2)))
    f.append(np.mean((np.array(s)-np.array(o))**2))
    print(np.mean((np.array(s)-np.array(o))**2))
    if i > num_eps - 10:
        plt.plot(o)
        plt.plot(s)
        #plt.plot(g)
        plt.show()

plt.plot(f)
#plt.show()

torch.save(model.state_dict(), "model_no_reg_3")
print("saving finished")

apm.reset()
h = 0
g = []
g_2 = []
for j in range(0, int(num_steps)):
    price = apm.get_current_price()
    delta = opm.compute_delta((num_steps - j) * dt, price)
    delta_action = delta - h

    data = torch.tensor(np.array([h, price, T - j * dt])).float()
    model_action = model(data)

    h = h + model_action.detach().numpy().item()
    apm.compute_next_price()

    g.append(model_action.detach().numpy().item())
    g_2.append(delta_action)

plt.plot(g)
plt.plot(g_2)
plt.show()