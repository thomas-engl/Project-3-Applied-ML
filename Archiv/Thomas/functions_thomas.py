import numpy as np
import matplotlib.pyplot as plt
import pytorch as torch
from heat_612.py import heat_nn


def plots_pinn_pde_complex_rhs(np_seed, torch_seed, times = [0.01,0.1,]):
    np.random.seed(np_seed)
    torch.manual_seed(torch_seed)
    layers = [32, 64, 128, 128, 1]
    #in this example: the output layer uses identity as activation func and all the hidden layers use tanh
    activations = [torch.tanh]*(len(layers)-1) + [None]
    dim=1
    kappa = 0.1
    u_0 = lambda x: torch.sin(torch.pi * x) + torch.sin(4 * torch.pi * x)
    rhs = lambda x, t: torch.sin(torch.pi * x)
    u_analytic = lambda x, t: (1 - 1 / (0.1 * torch.pi**2)) * torch.sin(torch.pi * x) * torch.exp(-torch.pi**2 * 0.1 * t
                        ) + torch.sin(4 * torch.pi * x) * torch.exp(- 16 * torch.pi**2 * 0.1 * t
                        ) + 1 / (0.1 * torch.pi**2) * torch.sin(torch.pi * x)

    pde_nn = heat_nn(layers, activations, dim, u_0, kappa, rhs, reg = 0)
    pde_nn.set_analytic_solution(u_analytic)
    N_colloc = 100

    x = np.linspace(0, 1, N_colloc)
    t = np.linspace(0, 0.2, N_colloc)

    xs, ts = np.meshgrid(x, t)

    xs = torch.tensor(xs, dtype=torch.float32).view(-1, 1)
    ts = torch.tensor(ts, dtype=torch.float32).view(-1, 1)

    x_colloc = torch.tensor(xs, dtype=torch.float32).view(-1,1)
    t_colloc = torch.tensor(ts, dtype=torch.float32).view(-1,1)

    pde_nn.x = [x_colloc]
    pde_nn.t = t_colloc

    # gives relatively good results (compared to other parameters, still bad though)
    pde_nn.train(lr=1e-2, weight_decay=0.0, epochs = 750, opt_time_scale =False, print_epochs=50)
    # LBFGS needs approximately 100 epochs, 30 iterations for kappa = 1
    # if kappa = 0.1, better choose more iterations, less epochs
    pde_nn.train_lbfgs(lr=1, opt_time_scale = False, epochs=10, max_iter=100, damping = False)

    for t in times:
        x_test = torch.linspace(0,1,100).view(-1,1)
        t_test = torch.tensor([[t]]*100)  # t=0.5
        u_pred = pde_nn.trial_solution(x_test, t_test).detach().numpy()

        #compare results with analytic solution

        x_np = x_test.numpy().flatten()  # convert to 1D array for plotting
        t_val = t
        u_analytic_val = pde_nn.u_analytic(x_test, t_test)

        # Plot the results
        plt.figure(dpi=150)
        plt.plot(x_np, u_pred, color='red', label='NN Prediction', linewidth=2)
        plt.plot(x_np, u_analytic_val, '--', color='yellow', label='Analytic Solution', linewidth=2)
        plt.xlabel(r'$x$')
        plt.ylabel(r'$u(x,t={})$'.format(t))
        plt.ylim(-0.6, 2.0)
        plt.legend()
        plt.show()

    ### error measured in L^2 and L^{\infty} norm
    L_2_err = pde_nn.L_2_error()
    L_infty_err = pde_nn.L_infty_error()
    print("L^2 error:        ", L_2_err)
    print("L_^{infty}_error: ", L_infty_err)