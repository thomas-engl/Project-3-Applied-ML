import torch
import matplotlib.pyplot as plt
import numpy as np
from heat_1d import heat_1d_nn

np.random.seed(238)
torch.manual_seed(301)

layers = [200, 100, 50, 1]
#in this example: the output layer uses identity as activation func and all the hidden layers use tanh
activations = [torch.tanh]*(len(layers)-1) + [None]
pde_nn = heat_1d_nn(layers, activations)

N_colloc = 10000
x_colloc = torch.rand(N_colloc, 1)
t_colloc = torch.rand(N_colloc, 1)
pde_nn.train(x_colloc, t_colloc, lr=1e-3, weight_decay=0, epochs =5000, printer=True)
#proper result for epochs =5000

x_test = torch.linspace(0,1,100).view(-1,1)
t_test = torch.tensor([[0.5]]*100)  # t=0.5
u_pred = pde_nn.trial_solution(x_test, t_test).detach().numpy()


#compare results with analytic solution

x_np = x_test.numpy().flatten()  # convert to 1D array for plotting
t_val = 0.5
u_analytic = np.sin(np.pi * x_np) * np.exp(-np.pi**2 * t_val)

# Plot the results
plt.figure(figsize=(8,5))
plt.plot(x_np, u_pred, label='NN Prediction', linewidth=2)
plt.plot(x_np, u_analytic, '--', label='Analytic Solution', linewidth=2)
plt.xlabel('x')
plt.ylabel('u(x,t=0.5)')
plt.title('Heat Equation: NN vs Analytic Solution')
plt.legend()
plt.grid(True)
plt.show()