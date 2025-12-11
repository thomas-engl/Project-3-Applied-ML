import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from heat_612 import heat_nn


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
    pde_nn.train(lr=1e-2, weight_decay=0.0, epochs = 750, opt_time_scale =True, print_epochs=50)
    # LBFGS needs approximately 100 epochs, 30 iterations for kappa = 1
    # if kappa = 0.1, better choose more iterations, less epochs
    pde_nn.train_lbfgs(lr=1, opt_time_scale = True, epochs=10, max_iter=100, damping = False)

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


def analyze_different_learning_rates_adam(np_seed, torch_seed, learning_rates, epochs):
    np.random.seed(np_seed)
    torch.manual_seed(torch_seed)
    #analyze different learning rate for Adam only
    #code for notizen time scale vs no time scale
    np.random.seed(23)
    torch.manual_seed(30)
    learning_rates = [0.0001, 0.001, 0.01, 0.1, 1]
    t_test = 0.2 * torch.rand(1000).view(-1,1)
    x_test = torch.rand(1000).view(-1,1)
    list_l2errors_train_lr = []
    list_linf_errors_train_lr = []
    list_l2errors_test_lr = []
    list_linf_errors_test_lr = []
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

    N_colloc = 100

    x = np.linspace(0, 1, N_colloc)
    t = np.linspace(0, 0.2, N_colloc)

    xs, ts = np.meshgrid(x, t)

    xs = torch.tensor(xs, dtype=torch.float32).view(-1, 1)
    ts = torch.tensor(ts, dtype=torch.float32).view(-1, 1)

    x_colloc = torch.tensor(xs, dtype=torch.float32).view(-1,1)
    t_colloc = torch.tensor(ts, dtype=torch.float32).view(-1,1)

    values = u_analytic(xs, ts)
    best_constant = values.mean().item()
    L_inf_best_constant = torch.max(torch.abs(values - best_constant))
    print("L inf best constant", L_inf_best_constant)

    for learning_rate in learning_rates:
        pde_nn = heat_nn(layers, activations, dim, u_0, kappa, rhs, reg = 0)
        pde_nn.set_analytic_solution(u_analytic)


        pde_nn.x = [x_colloc]
        pde_nn.t = t_colloc

        # gives relatively good results (compared to other parameters, still bad though)
        pde_nn.train(lr=learning_rate, weight_decay=0.0, epochs = epochs, opt_time_scale =False, print_epochs=0)
        list_l2errors_train_lr.append(pde_nn.L_2_error())
        list_linf_errors_train_lr.append(pde_nn.L_infty_error())
        list_l2errors_test_lr.append(pde_nn.L_2_error(x_test, t_test))
        list_linf_errors_test_lr.append(pde_nn.L_infty_error(x_test, t_test))
    
    print(f"{list_l2errors_train_lr=}")
    print(f"{list_linf_errors_train_lr=}")
    print(f"{list_l2errors_test_lr=}")
    print(f"{list_linf_errors_test_lr=}")


def epochs_plot(np_seed, torch_seed):
    #show the effect of combining the optimizers
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
    pde_nn.train(lr=1e-2, weight_decay=0.0, epochs = 750, opt_time_scale =True, save_losses = True, print_epochs=50)
    losses_adam_first = pde_nn.losses
    pde_nn.train_lbfgs(lr=1, opt_time_scale = True, epochs=10, max_iter=100, save_losses = True, print_epochs=1, damping = False)
    losses_lbfgs_further = pde_nn.losses
    # X-axis for Adam
    x_adam = np.arange(len(losses_adam_first))
    stretch_factor = 100
    # X-axis for LBFGS (stretched for visibility)
    x_lbfgs = len(losses_adam_first) + np.arange(len(losses_lbfgs_further)) * stretch_factor

    plt.figure(figsize=(12, 6))

    # Plot Adam
    plt.semilogy(x_adam, np.array(losses_adam_first), label="Adam", linewidth=2)

    # Plot LBFGS
    plt.semilogy(x_lbfgs, np.array(losses_lbfgs_further), label="L-BFGS", linewidth=2)

    # Mark transition point
    plt.axvline(len(losses_adam_first), color="gray", linestyle="--", alpha=0.6)
    plt.text(len(losses_adam_first), 
                min(min(losses_adam_first), min(losses_lbfgs_further))*1.1,
                "  switch to LBFGS",
                verticalalignment="bottom",
                color="gray")

    plt.xlabel("Iteration (Adam + LBFGS)")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

def analyze_regularization(np_seed, torch_seed, weights):
    np.random.seed(np_seed)
    torch.manual_seed(torch_seed)
    epochs_adam= 750
    epochs_lbfgs=10
    iter_lbfgs = 100
    MAX_RETRIES = 3          # run at most 3 times per weight
    THRESHOLD = 10        # any threshold you consider "too bad"
    weights = np.concatenate(([0.0], np.logspace(-4, 1, 5)))
    list_l2errors_train = []
    list_linf_errors_train = []
    list_l2errors_test = []
    list_linf_errors_test = []
    t_test = 0.2 * torch.rand(1000).view(-1,1)
    x_test = torch.rand(1000).view(-1,1)

    # === collocation points ===
    x = np.linspace(0, 1, 100)
    t = np.linspace(0, 0.2, 100)
    xs, ts = np.meshgrid(x, t)

    x_colloc = torch.tensor(xs, dtype=torch.float32).view(-1,1)
    t_colloc = torch.tensor(ts, dtype=torch.float32).view(-1,1)

    for w in weights:
        print(f"Testing weight decay: {w}")

        success = False

        for attempt in range(1, MAX_RETRIES + 1):
            print(f"  Attempt {attempt} for weight {w}")

            # === initialize model fresh ===
            layers = [32, 64, 128, 128, 1]
            activations = [torch.tanh]*(len(layers)-1) + [None]
            dim=1
            kappa = 0.1
            u_0 = lambda x: torch.sin(torch.pi * x) + torch.sin(4 * torch.pi * x)
            rhs = lambda x, t: 0
            
            u_analytic = lambda x, t: (
            torch.sin(torch.pi * x) * torch.exp(-kappa * (torch.pi**2) * t) +
            torch.sin(4 * torch.pi * x) * torch.exp(-kappa * (4*torch.pi)**2 * t)
            )


            pde_nn = heat_nn(layers, activations, dim, u_0, kappa, rhs, w)
            pde_nn.set_analytic_solution(u_analytic)

            
            pde_nn.x = [x_colloc]
            pde_nn.t = t_colloc

            # === training ===
            pde_nn.train(lr=1e-2, weight_decay=0.0, epochs=epochs_adam, opt_time_scale=True, print_epochs=0)
            pde_nn.train_lbfgs(lr=1, opt_time_scale=True, epochs=epochs_lbfgs, max_iter=iter_lbfgs, print_epochs=0)

            # === evaluate ===
            err = pde_nn.L_2_error(x_test, t_test)

            if err < THRESHOLD:
                print(f"  Success! L2 error = {err:.3e}")
                success = True
                break
            else:
                print(f"  Too large error: {err:.3e}, retrying...")

        if not success:
            print(f"   All retries failed for weight {w}. Using last result.")

        # === record final errors (after success or after retries) ===
        list_l2errors_train.append(pde_nn.L_2_error())
        list_linf_errors_train.append(pde_nn.L_infty_error())
        list_l2errors_test.append(pde_nn.L_2_error(x_test, t_test))
        list_linf_errors_test.append(pde_nn.L_infty_error(x_test, t_test))
    print(f"{list_l2errors_train=}")
    print(f"{list_linf_errors_train=}")
    print(f"{list_l2errors_test=}")
    print(f"{list_linf_errors_test=}")


def analyze_activation_funcs(np_seed, torch_seed):
    np.random.seed(np_seed)
    torch.manual_seed(torch_seed)
    layers = [64, 64, 64, 64, 1]
    #activations = [torch.tanh]*(len(layers)-1) + [None]
    dim=1
    kappa = 0.1
    u_0 = lambda x: torch.sin(torch.pi * x) + torch.sin(4 * torch.pi * x)
    rhs = lambda x, t: 0

    u_analytic = lambda x, t: (
    torch.sin(torch.pi * x) * torch.exp(-kappa * (torch.pi**2) * t) +
    torch.sin(4 * torch.pi * x) * torch.exp(-kappa * (4*torch.pi)**2 * t)
    )

    activations_list = [[torch.tanh]*(len(layers)-1) + [None], [torch.relu]*(len(layers)-1) + [None], [torch.sigmoid]*(len(layers)-1) + [None]]
    epochs_adam= 750
    epochs_lbfgs=10
    iter_lbfgs = 100
    MAX_RETRIES = 3          # run at most 3 times per weight
    THRESHOLD = 10        # any threshold you consider "too bad"
    weights = np.concatenate(([0.0], np.logspace(-4, 1, 5)))
    list_l2errors_train_adam = []
    list_linf_errors_train_adam = []
    list_l2errors_test_adam = []
    list_linf_errors_test_adam = []
    list_l2errors_train_lbfgs = []
    list_linf_errors_train_lbfgs = []
    list_l2errors_test_lbfgs = []
    list_linf_errors_test_lbfgs = []
    t_test = 0.2 * torch.rand(1000).view(-1,1)
    x_test = torch.rand(1000).view(-1,1)

    # === collocation points ===
    x = np.linspace(0, 1, 100)
    t = np.linspace(0, 0.2, 100)
    xs, ts = np.meshgrid(x, t)

    x_colloc = torch.tensor(xs, dtype=torch.float32).view(-1,1)
    t_colloc = torch.tensor(ts, dtype=torch.float32).view(-1,1)

    for activations in activations_list:
        print("Testing ", activations[0])

        success = False

        for attempt in range(1, MAX_RETRIES + 1):
            print(f"  Attempt {attempt} for", activations[0])

            # === initialize model fresh ===


            pde_nn = heat_nn(layers, activations, dim, u_0, kappa, rhs, 0)
            pde_nn.set_analytic_solution(u_analytic)

            
            pde_nn.x = [x_colloc]
            pde_nn.t = t_colloc

            # === training ===
            pde_nn.train(lr=1e-2, weight_decay=0.0, epochs=epochs_adam, opt_time_scale=False, print_epochs=0)
            adam_train_l2 = pde_nn.L_2_error()
            adam_train_linf = pde_nn.L_infty_error()
            adam_test_l2 = pde_nn.L_2_error(x_test, t_test)
            adam_test_linf = pde_nn.L_infty_error(x_test, t_test)
            pde_nn.train_lbfgs(lr=1, opt_time_scale=False, epochs=epochs_lbfgs, max_iter=iter_lbfgs, print_epochs=0)

            # === evaluate ===
            err = pde_nn.L_2_error(x_test, t_test)

            if err < THRESHOLD:
                print(f"  Success! L2 error = {err:.3e}")
                success = True
                break
            else:
                print(f"  Too large error: {err:.3e}, retrying...")

        if not success:
            print(f"   All retries failed. Using last result.")

        # === record final errors (after success or after retries) ===
        list_l2errors_train_adam.append(adam_train_l2)
        list_linf_errors_train_adam.append(adam_train_linf)
        list_l2errors_test_adam.append(adam_test_l2)
        list_linf_errors_test_adam.append(adam_test_linf)

        list_l2errors_train_lbfgs.append(pde_nn.L_2_error())
        list_linf_errors_train_lbfgs.append(pde_nn.L_infty_error())
        list_l2errors_test_lbfgs.append(pde_nn.L_2_error(x_test, t_test))
        list_linf_errors_test_lbfgs.append(pde_nn.L_infty_error(x_test, t_test))
    print(f"{list_l2errors_train_lbfgs=}")
    print(f"{list_linf_errors_train_lbfgs=}")
    print(f"{list_l2errors_test_lbfgs=}")
    print(f"{list_linf_errors_test_lbfgs=}")

    print(f"{list_l2errors_train_adam=}")
    print(f"{list_linf_errors_train_adam=}")
    print(f"{list_l2errors_test_adam=}")
    print(f"{list_linf_errors_test_adam=}")


