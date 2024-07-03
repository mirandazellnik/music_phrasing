import sys
import os
import argparse
import json
import random

#import tensorflow
#from tensorflow import keras
#import keras_tuner
import pickle
import numpy as np

from reservoirpy.observables import mse, rsquare
from reservoirpy.nodes import Reservoir, Ridge # type: ignore
from reservoirpy.hyper import plot_hyperopt_report # type: ignore
from reservoirpy.hyper import research # type: ignore
from util.load_data import prepare_dataset
from util.reservoir_model import create_model, load_model, store_model


data_path = "/stash/tlab/theom_intern/midi_data/asap-dataset-processed/shifted_by_piece.json"
metadata_path = "/stash/tlab/theom_intern/midi_data/asap-dataset-master/metadata.csv"

argParser = argparse.ArgumentParser()
argParser.add_argument("name", nargs="?", help="Name of this run, for logging, model saving, etc.", type=str)
argParser.add_argument("cpu_name", nargs="?", help="Name of cpu this file is running on, for logging, model saving, etc.", type=str)
argParser.add_argument("-t", "--no-train", help="Don't train a new model, instead load the existing model.", action=argparse.BooleanOptionalAction)
argParser.add_argument("-T", "--tune", help="Tune the model with hyperband.", action=argparse.BooleanOptionalAction)
argParser.add_argument("-g", "--goal", help="Goal variable")
args = argParser.parse_args()

assert args.name
save_name = args.name
cpu_name = args.cpu_name
goal = args.goal
if not goal:
    goal = "Micro"
assert goal in ["Micro", "Len_P"]

if args.tune:
    with open(f"/stash/tlab/theom_intern/distributed_reservoir_runs/{save_name}/{cpu_name}_hp_search/completion.txt", "w+") as f:
        f.write("0")

# Micro data (current and -1): Note, Exact_Lower, Exact_Higher, Motion
# Micro data (current only): Len_M, W.5, B.1, B.5, A.1, A.5, W.5
# Macro data: Time, B1, B2, B4, W.5

print("Preparing dataset...")
if goal == "Micro":
    """trd, trt, vad, vat, ted, tet = prepare_dataset(
        data_path, metadata_path,
        ["Note", "Exact_L", "Len/BPM", "Exact_H", "Motion", "Micro"],
        ["Len_M", "Melodic_Charge", "W.50", "B.10", "B.50", "A.10", "A.50", "W.50"],
        ["Micro"]
    )"""
    trd, trt, vad, vat, ted, tet = prepare_dataset(
        data_path, metadata_path,
        ["Note", "Exact_L", "Len/BPM"],
        ["Len_M", "Melodic_Charge", "Micro"],
        ["Micro"],
        train_by_piece=True
    )
elif goal == "Len_P":
    trd, trt, vad, vat, ted, tet = prepare_dataset(
        data_path, metadata_path,
        ["Len_M", "D_A", "Len/BPM", "Len_Ratio"],
        ["Exact", "Len_P", "Micro", "B.50", "B2.0", "A.50", "A2.0", "W.50", "W2.0"],
        ["Len_P"]
    )

"""
hyperopt_config = {
    "exp": f"/stash/tlab/theom_intern/hyper_exp_data/{save_name}",    # the experimentation name
    "hp_max_evals": 300,              # the number of differents sets of parameters hyperopt has to try
    "hp_method": "random",            # the method used by hyperopt to chose those sets (see below)
    "seed": 42,                       # the random state seed, to ensure reproducibility
    "instances_per_trial": 5,         # how many random ESN will be tried with each sets of parameters
    "hp_space": {                     # what are the ranges of parameters explored
        "N": ["choice", 10, 20, 30, 50, 100, 200, 400, 600, 1000],             # the number of neurons is fixed to 500
        "sr": ["loguniform", 1e-2, 10],   # the spectral radius is log-uniformly distributed between 1e-2 and 10
        "lr": ["loguniform", 1e-3, 1],    # idem with the leaking rate, from 1e-3 to 1
        "input_scaling": ["choice", 1.0], # the input scaling is fixed
        "ridge": ["loguniform", 1e-8, 1e1],        # and so is the regularization parameter.
        "seed": ["choice", 1234]          # an other random seed for the ESN initialization
    }
}

with open(f"/stash/tlab/theom_intern/hp_model_configs/{save_name}.config.json", "w+") as f:
    json.dump(hyperopt_config, f)
"""

test_data_vel, test_goals_vel = prepare_dataset(
    data_path, metadata_path,
    ["Note", "Exact_L", "Len/BPM"],
    ["Len_M", "Melodic_Charge", "Micro"],
    ["Micro"],
    test_data_only=True
)

def objective(dataset, config, *, N, sr, lr, input_scaling, ridge, seed):
    global current_run
    
    trd, trt, vad, vat = dataset

    trd_keys = list(trd.keys())
    random.Random(4).shuffle(trd_keys)
    trd = [trd[k] for k in trd_keys]
    trt = [trt[k] for k in trd_keys]

    trd, trt, = map(lambda abc: [np.squeeze(one_piece.to_numpy()) for one_piece in abc], [trd, trt])

    trt = [i.reshape(-1, 1) for i in trt]

    print(trd[0].shape, trt[0].shape, vad.shape, vat.shape)

    instances = config['instances_per_trial']

    # the seed should change between each trial to prevent bias
    trial_seed = seed

    losses = []
    r2s = []
    for i in range(instances):
        model = create_model(input_scaling, N, sr, lr, ridge, trial_seed)

        model.fit(trd, trt)

        pieces = list(test_data_vel.keys())
        print(len(pieces))

        loss = 0
        r2 = 0

        for i, piece in enumerate(pieces):

            piece_data_vel = test_data_vel[piece]

            vad = np.squeeze(piece_data_vel.to_numpy())
            #vad = piece_data_vel
            piece_len = vad.shape[0]

            predictions = model.run(vad, reset=True)


            """
            predictions = []

            for row in range(len(vad)):
                if goal == "Micro" and row > 0:
                    vad.loc[row, "-1_Micro"] = float(out) 
                inputs = vad.loc[row]
                #if row < 3:
                #    print(inputs)

                inputs = np.squeeze(inputs.to_numpy())
                #if row < 3:
                #    print(inputs)

                out = model(inputs)[0]
                predictions.append(out)

                #de = f"{ted.iloc[row]['Len_M']}"
                #print(out)
            """

            a, b = [predictions[j][0] for j in range(len(predictions))], [test_goals_vel[piece].iloc[j]["Micro"] for j in range(len(predictions))]
            
            if i == 8:
                for j in range(len(predictions)):
                    print(f"{a[j]}\t{b[j]}")
            
            loss += mse(a, b)
            r2 += rsquare(b, a)

            print(f"Piece {i}: {mse(a, b)}")

            #print(f"Piece {i}: {loss}")
        
        loss /= len(pieces)
        r2 /= len(pieces)

        print(f"All pieces: {loss}")

        losses.append(loss)
        r2s.append(r2)

        trial_seed += 1

        if args.tune:
            current_run += 1

            with open(f"/stash/tlab/theom_intern/distributed_reservoir_runs/{save_name}/{cpu_name}_hp_search/completion.txt", "w+") as f:
                f.write(f"{current_run}")

    if args.tune:
        with open(f"/stash/tlab/theom_intern/distributed_reservoir_runs/{save_name}/{cpu_name}_hp_search/{cpu_name}_all_hps.txt", "a") as f:
            f.write(f"\n{N}\t{sr}\t{lr}\t{ridge}\t{input_scaling}\t{np.mean(losses)}")
    else:
        print("Saving model...")
        pickle.dump(model, open(f"/stash/tlab/theom_intern/res_models/{save_name}.p", "wb" ) )
    
    return {'loss': np.mean(losses),
            'r2': np.mean(r2s)}


if args.tune:
    with open(f"/stash/tlab/theom_intern/distributed_reservoir_runs/{save_name}/{cpu_name}_hp_search/{cpu_name}.config.json", 'r') as f:
        conf = json.load(f)
    current_run = 0
    

    best = research(objective, [trd, trt, vad, vat], f"/stash/tlab/theom_intern/distributed_reservoir_runs/{save_name}/{cpu_name}_hp_search/{cpu_name}.config.json")
    with open(f"/stash/tlab/theom_intern/distributed_reservoir_runs/{save_name}/{cpu_name}_hp_search/{cpu_name}_best_hps.txt", "a") as f:
        f.write(str(best))
    fig = plot_hyperopt_report(f"/stash/tlab/theom_intern/distributed_reservoir_runs/{save_name}", ("lr", "sr", "ridge"), metric="loss")
    fig.savefig("/stash/tlab/theom_intern/figure1.png")
    #fig.show(block=True)
elif not args.no_train:
    print(objective([trd,trt,vad,vat], {"instances_per_trial":1}, N=1000, sr=1.028798288646009, lr=0.4809398365604778, ridge=4.99994379801419, input_scaling=1.0, seed=1234))
if True:
    pass
elif not args.no_train:
    models = []
    for i in range(1):
        print(f"----------------------------------- MODEL {i+1} -----------------------------------")
        model = create_model()
        #reservoir, readout = Reservoir(100, lr=0.2, sr=1.0), Ridge(ridge=1e-3)
        trd, trt, vad = map(lambda abc: np.squeeze(abc.to_numpy()), [trd, trt, vad])

        trt = trt.reshape(-1, 1)

        print(trd.shape, trt.shape)
        #model.fit(trd, trt)
        model.fit(trd, trt)
        #model.fit(trd, trt, validation_data=(vad, vat), epochs=100, batch_size=32, verbose=2, callbacks=[tensorboard] if save_name else [])
        #models.append(model)
    #model.save(f"/stash/tlab/theom_intern/models/{save_name}/{goal}")   
    test_data_vel, test_goals_vel = prepare_dataset(
        data_path, metadata_path,
        ["Note", "Exact_L", "Len/BPM", "Micro"],
        ["Len_M", "Melodic_Charge"],
        ["Micro"],
        test_data_only=True
    )

    pieces = list(test_data_vel.keys())
    print(len(pieces))

    total_mse = 0
    no_model_mse = 0
    total_mse_len = 0

    for i, piece in enumerate(pieces):


        piece_data_vel = test_data_vel[piece]

        vad = np.squeeze(piece_data_vel.to_numpy())
        piece_len = vad.shape[0]

        predictions = model.run(vad, reset=True)


        piece_mse = 0

        for j in range(len(predictions)):
            piece_mse += (predictions[i][0] - test_goals_vel[piece].iloc[i]["Micro"])**2
            no_model_mse += (test_goals_vel[piece].iloc[i]["Micro"])**2
            #print(f'{predictions[i][0]}\t{test_goals_vel[piece].iloc[i]["Micro"]}')
        total_mse += piece_mse
        total_mse_len += piece_len
        piece_mse /= piece_len
        print(f"Piece {i}: {piece_mse}")
    print(f"All pieces: {total_mse / total_mse_len}")
    print(f"No model: {no_model_mse / total_mse_len}")

    # storeModel(model)

else:
    pass
    # model_to_test = load_model(name)

# test model

