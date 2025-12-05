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
    def __init__(self, layers, activations, dim, u_0, kappa, rhs):
        self.u_analytic = None
        self.net = FFNN(layers, activations, dim=dim)
        self.dim = dim
        self.u_0 = u_0
        self.kappa = kappa
        self.rhs = rhs
        self.time_scale = nn.Parameter(torch.tensor(1.0))
    
    def trial_solution(self, *args):
        #trial_solution(x,t), different order breaks it
        L=1
        t = args[-1]
        x = list(args[:-1])      
        x_scaled = [2*xx - 1 for xx in x]
        t_scaled = torch.exp(self.time_scale) * t
        N = self.net(torch.cat(x_scaled + [t_scaled], dim=1))
        trial_sol = t * N
        for d in x:
            trial_sol *= d * (L - d)
        return torch.exp(-t) * self.u_0(*x) + trial_sol  

    def pde_residual(self, x, t):
        t.requires_grad = True
        for d in x:
            d.requires_grad = True
        
        u = self.trial_solution(*x, t)    
        u_xx_sum = 0
        for d in x:
            u_d = torch.autograd.grad(u, d, grad_outputs=torch.ones_like(u), create_graph=True, allow_unused=True)[0]
            # Use ones_like(u_d) here so grad_outputs shape matches u_d
            u_dd = torch.autograd.grad(u_d, d, grad_outputs=torch.ones_like(u_d), create_graph=True, allow_unused=True)[0]
            u_xx_sum += u_dd
        
        u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True, allow_unused=True)[0]
        return u_t - self.kappa * u_xx_sum - self.rhs  # Heat equation: u_t - kappa * u_xx = rhs

    def loss_fn(self, x, t):
        f = self.pde_residual(x, t)
        # print("pde residual computed")
        return torch.mean(f**2)

    def set_analytic_solution(self, u_analytic):
        self.u_analytic = u_analytic
    
    def mse(self, x, t):
        if self.u_analytic is None:
            raise ValueError("Analytic solution not set. Use set_analytic_solution method to set it.")

        with torch.no_grad():
            u_pred = self.trial_solution().detach().numpy()
            u_exact = self.u_analytic(*x, t).detach().numpy()
            mse = np.mean((u_pred - u_exact)**2)
        
        return mse

    def train(self, x_colloc, t_colloc, lr, weight_decay, epochs, opt_time_scale =True, print_epochs = 500):
        #weight_decay is for Adam not the same as L2 regularization but it is recommended to use
        if opt_time_scale:
            parameters = list(self.net.parameters()) + [self.time_scale]
        else:
            parameters = list(self.net.parameters())
        
        optimizer = torch.optim.Adam(parameters, lr=lr, weight_decay = weight_decay)
        for epoch in range(epochs):
            optimizer.zero_grad()
            loss = self.loss_fn(x_colloc, t_colloc)
            loss.backward()
            optimizer.step()
        
        if print_epochs != 0:
            if epoch % print_epochs == 0 or epoch == epochs - 1:
                if self.u_analytic is not None:
                    print(f"Epoch {epoch}, Loss: {loss.item():.6f}, MSE: {self.mse(x, t):.6f}")
                else:
                    print(f"Epoch {epoch}, Loss: {loss.item():.6f}")


    def train_lbfgs(self, x_colloc, t_colloc, lr, opt_time_scale = True, epochs=50, max_iter = 50):
        if opt_time_scale:
            parameters = list(self.net.parameters()) + [self.time_scale]
        else:
            parameters = list(self.net.parameters())        
        optimizer = torch.optim.LBFGS(parameters, lr=lr, max_iter = max_iter)

        for epoch in range(epochs):
            def closure():
                optimizer.zero_grad()
                loss = self.loss_fn(x_colloc, t_colloc)
                loss.backward()
                return loss

            optimizer.step(closure)
            print(f"LBFGS Epoch {epoch+1}, Loss: {closure().item():.6f}")

