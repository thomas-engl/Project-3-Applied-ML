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
        self.u_0 = u_0
    
    def trial_solution(self, *args):
        if args:
            t = args[-1]
            x = args[:-1]
        else:
            t = self.t
            x = self.x
        L = 1
        N = self.net(torch.cat(x + (t,), dim=1))
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
        self.args = args
        self.t = args[-1]
        self.x = args[:-1]

    def train(self, lr, weight_decay, epochs, print_epochs=-1):
        optimizer = torch.optim.Adam(self.net.parameters(), lr=lr, weight_decay=weight_decay)
        for epoch in range(epochs):
            optimizer.zero_grad()
            loss = self.loss_fn()
            loss.backward()
            optimizer.step()

            if epoch % print_epochs == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch}, Loss: {loss.item():.6f}")



class heat_1d_nn():
    def __init__(self, layers, activations):
        self.net = FFNN(layers, activations, dim=1)

    # Trial solution for 1d heat equation
    def trial_solution(self, x, t):
        L=1
        N = self.net(torch.cat([x, t], dim=1))
        f = torch.sin(torch.pi * x)
        return torch.exp(-10 * t**2) * f + x * (L - x) * t * N

    # Residuals for 1d heat equation
    def pde_residual(self, x, t):
        x.requires_grad = True
        t.requires_grad = True

        u = self.trial_solution(x, t)

        u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]

        return u_t - u_xx  # Heat equation: u_t - u_xx = 0

    # calculate loss
    def loss_fn(self, x_colloc, t_colloc):
        f = self.pde_residual(x_colloc, t_colloc)
        return torch.mean(f**2)

    def train(self, x_colloc, t_colloc, lr, weight_decay, epochs, print_epochs = 500):
        #weight_decay is for Adam not the same as L2 regularization but it is recommended to use
        optimizer = torch.optim.Adam(self.net.parameters(), lr=lr, weight_decay = weight_decay)
        for epoch in range(epochs):
            optimizer.zero_grad()
            loss = self.loss_fn(x_colloc, t_colloc)
            loss.backward()
            optimizer.step()

            if epoch % print_epochs == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch}, Loss: {loss.item():.6f}")



class heat_2d_nn():
    """
    solve the 2d Heat equation
    
                u_t - \Delta u = f
    
    on \Omega = (0, 1)^2 with homogeneous Dirichlet boundary conditions and
    u(0, - ) = u_0
    """
    def __init__(self, layers, activations):
        self.net = FFNN(layers, activations, dim=2)

    # Trial solution for 1d heat equation
    def trial_solution(self, x, y, t):
        
        N = self.net(torch.cat([x, y, t], dim=1))
        # initial condition
        u_0 = torch.sin(torch.pi * x) * torch.sin(torch.pi * y) + torch.sin(
                2 * torch.pi * x) * torch.sin(4 * torch.pi * y)
        return u_0 + x * (1 - x) * y * (1 - y) * t * N

    # Residuals for 1d heat equation
    def pde_residual(self, x, y, t):
        x.requires_grad = True
        y.requires_grad = True
        t.requires_grad = True

        u = self.trial_solution(x, y, t)

        u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        
        u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]

        return u_t - u_xx - u_yy # Heat equation: u_t - \Delta u = 0

    # calculate loss
    def loss_fn(self, x_colloc, y_colloc, t_colloc):
        f = self.pde_residual(x_colloc, y_colloc, t_colloc)
        return torch.mean(f**2)

    def train(self, x_colloc, y_colloc, t_colloc, lr, weight_decay, epochs, print_epochs = 500):
        #weight_decay is for Adam not the same as L2 regularization but it is recommended to use
        optimizer = torch.optim.Adam(self.net.parameters(), lr=lr, weight_decay = weight_decay)
        for epoch in range(epochs):
            optimizer.zero_grad()
            loss = self.loss_fn(x_colloc, y_colloc, t_colloc)
            loss.backward()
            optimizer.step()

            if epoch % print_epochs == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

