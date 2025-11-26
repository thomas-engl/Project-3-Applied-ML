import torch
import torch.nn as nn
import numpy as np


class FFNN(nn.Module):
    def __init__(self, layers, activations, dim):
        super().__init__()

        # Expect one activation per layer (including input layer and output layer)
        assert len(activations) == len(layers), (
            "You must provide one activation per layer (including output)."
        )
        # Linear layers (between each pair of layers) excl. input layer
        linears = [nn.Linear(layers[i], layers[i+1]) for i in range(len(layers)-1)]

        # input layer
        input_layer = [nn.Linear(dim+1, layers[0])]
        self.linears = nn.ModuleList(input_layer + linears)

        # Store activations, allow None for identity
        self.activations = []
        for act in activations:
            if act is None:
                self.activations.append(lambda x: x)
            else:
                self.activations.append(act)

    def forward(self, x):
        # Apply linear layers and their activations
        for linear, activation in zip(self.linears, self.activations):
            x = activation(linear(x))

        return x


class heat_nn():
    def __init__(self, layers, activations, dim, u_0):
        self.net = FFNN(layers, activations, dim=dim)
        self.dim = dim
        self.u_0 = u_0
    
    def trial_solution(self, *args):
        if args:
            t = args[-1]
            x = list(args[:-1])
        else:
            t = self.t
            x = self.x
        L = 1
        N = self.net(torch.cat(x + [t], dim=1))
        trial_sol = t * N
        for d in x:
            trial_sol *= d * (L - d)
        return (1-t) * self.u_0(*x) + trial_sol
    
    def pde_residual(self):

        self.t.requires_grad = True
        for d in self.x:
            d.requires_grad = True
        
        u = self.trial_solution()

        u_t = torch.autograd.grad(u, self.t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_xx_sum = 0
        for d in self.x:
            u_d = torch.autograd.grad(u, d, grad_outputs=torch.ones_like(u), create_graph=True)[0]
            u_dd = torch.autograd.grad(u_d, d, grad_outputs=torch.ones_like(u), create_graph=True)[0]
            u_xx_sum += u_dd
        
        return u_t - u_xx_sum  # Heat equation: u_t - \Delta u = 0
    
    def loss_fn(self):
        f = self.pde_residual()
        return torch.mean(f**2)
    
    def set_data(self, *args):
        if len(args) == 1:
            args = [args[0]] * (self.dim + 1)
        elif len(args) != self.dim + 1:
            raise ValueError(f"Expected {self.dim + 1} arguments (one per dimension + time) or a single argument to be used for all, got {len(args)}.")

        x = []
        for d in args[:-1]:
            x.append(np.linspace(0, 1, d))
        t = np.linspace(0, 1, args[-1])

        mesh = np.meshgrid(*x, t)

        xs = mesh[:-1]
        ts = mesh[-1]

        x_colloc_list = []
        for d in range(self.dim):
            x_colloc_list.append(torch.tensor(xs[d], dtype=torch.float32).view(-1,1))
        t_colloc = torch.tensor(ts, dtype=torch.float32).view(-1,1)

        self.x = x_colloc_list
        self.t = t_colloc
        self.args = x_colloc_list + [t_colloc]

    def train(self, lr, weight_decay, epochs, print_epochs=-1):
        optimizer = torch.optim.Adam(self.net.parameters(), lr=lr, weight_decay=weight_decay)
        for epoch in range(epochs):
            optimizer.zero_grad()
            loss = self.loss_fn()
            loss.backward()
            optimizer.step()

            if epoch % print_epochs == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch}, Loss: {loss.item():.6f}")
