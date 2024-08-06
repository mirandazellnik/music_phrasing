import sys
import os
import argparse

import tensorflow
from tensorflow import keras
import keras_tuner
import pickle
import numpy as np
from functools import wraps

from reservoirpy.nodes import Reservoir, Ridge, ESN # type: ignore


def create_model(input_scaling, N, sr, lr, ridge, seed, rc_connectivity=0.1, input_connectivity=0.1, activation_func='relu'):
        
    reservoir = Reservoir(
            units=N,
            sr=sr,
            lr=lr,
            input_scaling=input_scaling,
            seed=seed,
            rc_connectivity=rc_connectivity,
            input_connectivity=input_connectivity,
            activation=activation_func
        )

    print('units: ', N)
    print('sr: ', sr)
    print('lr: ', lr)
    print('input scaling: ', input_scaling)
    print('seed: ', seed)
    print('rc_connectivity: ', rc_connectivity)
    print('input connectivity: ', input_connectivity)
    print('activation: ', activation_func)
    print('ridge: ', ridge)

    #reservoirs = [Reservoir(100, lr=lr, sr=sr) for i in num_inputs]
    #print(f"{len(reservoirs)} Res[]s")
    ridge = Ridge(ridge=ridge)

    #esn = reservoir >> ridge
    esn = ESN(reservoir=reservoir, readout=ridge, workers=-1, feedback=False)

    return esn

def store_model(save_name, model, subname="main"):
    # Its important to use binary mode
    os.makedirs(os.path.dirname(f"/stash/tlab/theom_intern/res_models/{save_name}/{subname}/model.pkl"), exist_ok=True)
    dbfile = open(f"/stash/tlab/theom_intern/res_models/{save_name}/{subname}/model.pkl", 'ab')
    # source, destination
    pickle.dump(model, dbfile)                    
    dbfile.close()
 
def load_model(save_name, subname="main"):
    # for reading also binary mode is important
    dbfile = open(f"/stash/tlab/theom_intern/res_models/{save_name}/{subname}/model.pkl", 'rb')    
    model = pickle.load(dbfile)
    return model