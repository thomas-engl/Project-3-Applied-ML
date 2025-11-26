import torch
import torch.nn as nn
import numpy as np

class FFNN(nn.Module):
    def __init__(self, layers, activations):
        super().__init__()

        # Expect one activation per layer (including input layer and output layer)
        assert len(activations) == len(layers), (
            "You must provide one activation per layer (including output)."
        )
        # Linear layers (between each pair of layers) excl. input layer
        linears = [nn.Linear(layers[i], layers[i+1]) for i in range(len(layers)-1)]

        # input layer
        input_layer = [nn.Linear(2, layers[0])]
        self.linears = nn.ModuleList(input_layer + linears)

        # Store activations, allow None for identity
        self.activations = [lambda x: x]
        for act in activations:
            if act is None:
                self.activations.append(lambda x: x)
            else:
                self.activations.append(act)

    def forward(self, x):
        # Apply activation for the input layer first
        x = self.activations[0](x)

        # Apply linear layers and their activations
        for linear, activation in zip(self.linears, self.activations[1:]):
            x = activation(linear(x))

        return x


class heat_1d_nn():
    def __init__(self, layers, activations):
        self.net = FFNN(layers, activations)

    # Trial solution for 1d heat equation
    def trial_solution(self, x, t):
        L=1
        N = self.net(torch.cat([x, t], dim=1))
        f = torch.sin(torch.pi * x)
        return (1 - t) * f + x * (L - x) * t * N

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

    def train(self, x_colloc, t_colloc, lr, weight_decay, epochs, printer = True):
        #weight_decay is for Adam not the same as L2 regularization but it is recommended to use
        optimizer = torch.optim.Adam(self.net.parameters(), lr=lr, weight_decay = weight_decay)
        for epoch in range(epochs):
            optimizer.zero_grad()
            loss = self.loss_fn(x_colloc, t_colloc)
            loss.backward()
            optimizer.step()

            if epoch % 500 == 0 and printer:
                print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

