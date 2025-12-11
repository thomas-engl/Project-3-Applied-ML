"""
def analyze_weight_decay(np_seed, torch_seed, weights, epochs_adam, epochs_lbfgs, iter_lbfgs):

    np.random.seed(np_seed)
    torch.manual_seed(torch_seed)
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
            rhs = lambda x, t: torch.sin(torch.pi * x)
            u_analytic = lambda x, t: (1 - 1/(0.1*torch.pi**2))*torch.sin(torch.pi*x)*torch.exp(-torch.pi**2*0.1*t) \
                                    + torch.sin(4*torch.pi*x)*torch.exp(-16*torch.pi**2*0.1*t) \
                                    + 1/(0.1*torch.pi**2)*torch.sin(torch.pi*x)

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
"""