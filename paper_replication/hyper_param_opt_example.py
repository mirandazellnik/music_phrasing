import json

import numpy as np
import matplotlib.pyplot as plt

from reservoirpy.observables import nrmse, rsquare # type: ignore
from reservoirpy.hyper import plot_hyperopt_report # type: ignore
from reservoirpy.hyper import research # type: ignore

from model import model_build, model_train, model_predict


hyperopt_config = {
    "exp": "hyperopt-mackey_glass",    # the experimentation name
    "hp_max_evals": 200,              # the number of differents sets of parameters hyperopt has to try
    "hp_method": "random",            # the method used by hyperopt to chose those sets (see below)
    "seed": 42,                       # the random state seed, to ensure reproducibility
    "instances_per_trial": 5,         # how many random ESN will be tried with each sets of parameters
    "hp_space": {                     # what are the ranges of parameters explored
        "N": ["choice", 20],             # the number of neurons is fixed to 500
        "sr": ["loguniform", 1e-2, 10],   # the spectral radius is log-uniformly distributed between 1e-2 and 10
        "lr": ["loguniform", 1e-3, 1],    # idem with the leaking rate, from 1e-3 to 1
        "input_scaling": ["choice", 1.0], # the input scaling is fixed
        "ridge": ["loguniform", 1e-8, 1e1],        # and so is the regularization parameter.
        "seed": ["choice", 1234]          # an other random seed for the ESN initialization
    }
}

# we precautionously save the configuration in a JSON file
# each file will begin with a number corresponding to the current experimentation run number.
#with open(f"hp_model_configs/{hyperopt_config['exp']}.config.json", "w+") as f:
 #   json.dump(hyperopt_config, f)


# Objective functions accepted by ReservoirPy must respect some conventions:
#  - dataset and config arguments are mandatory, like the empty '*' expression.
#  - all parameters that will be used during the search must be placed after the *.
#       - they also must be named to correspond to the key names in the config file. If the spectral radius is named sr in the 
#         config file, it can't be named spectral_radius as parameter in the function declaration
#  - the function must return a dict with at least a 'loss' key containing the result
# of the loss function. You can add any additional metrics or information with other 
# keys in the dict. See hyperopt documentation for more informations.
def objective(dataset, config, *, N, sr, lr, input_scaling, ridge, seed):
    X_train, Y_train, X_test, Y_test = dataset

    instances = config['instances_per_trial']

    # the seed should change between each trial to prevent bias
    trial_seed = seed

    losses = []
    r2s = []
    for i in range(instances):
        model = model_build(N, 1, lr, input_scaling, sr, ridge, seed)

        model_train(model, X_train, Y_train, 0)
        predictions = model_predict(model, X_test)

        loss = nrmse(Y_test, predictions, norm_value=np.ptp(X_train))
        r2 = rsquare(Y_test, predictions)

        losses.append(loss)
        r2s.append(r2)

        trial_seed += 1
    
    return {'loss': np.mean(losses),
            'r2': np.mean(r2s)}


def find_best_params(dataset):
    return research(objective, dataset, f"hp_model_configs/{hyperopt_config['exp']}.config.json")