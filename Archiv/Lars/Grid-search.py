### Cell 1
import time
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid
from heat import heat_nn


### Cell 2
layers = []
for i in range(2, 5):
    for params in ParameterGrid({str(j): [50, 100, 200, 300] for j in range(i)}):
        layers.append([params[str(j)] for j in range(i)] + [1])

param_grid = {
    "layers": layers,
    "lr": [1e-3, 1e-4],
    "data points x": [10, 50, 100],
    "data points t": [10, 50, 100],
}


### Cell 3
results = pd.DataFrame(columns=["params", "epochs", "loss", "mse"])
start_time = time.time()

for idx, params in enumerate(ParameterGrid(param_grid)):
    actications = [torch.tanh for _ in range(len(params["layers"])-1)] + [None]
    model = heat_nn(params["layers"], actications, 1, lambda x: torch.sin(torch.pi * x))
    model.set_data(params["data points x"], params["data points t"])
    model.set_analytic_solution(lambda x, t: np.sin(np.pi * x) * np.exp(-np.pi**2 * t))
 
    for i in range(30):
        model.train(lr=params["lr"], weight_decay=0, epochs=100)
        loss = model.loss_fn()
        mse = model.mse()
        results = pd.concat([results, pd.DataFrame([{
            "params": params,
            "epochs": (i+1)*100,
            "loss": loss,
            "mse": mse
        }])], ignore_index=True)
    
    t = int(time.time() - start_time)
    print(f"Completed {idx+1} out of {len(ParameterGrid(param_grid))} configurations. Time elapsed: {t//3600}h {(t%3600)//60}m {t%60}s")



### Cell 4
with open("heat_nn_hyperparameter_tuning_results.pkl", "wb") as f:
    pd.to_pickle(results, f)