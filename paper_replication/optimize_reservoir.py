import sys
import os
import argparse
import json

#import tensorflow
#from tensorflow import keras
#import keras_tuner
import pickle
import numpy as np

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
        ["Note", "Exact_L", "Len/BPM", "Micro"],
        ["Len_M", "Melodic_Charge"],
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

test_data_vel, test_goals_vel = prepare_dataset(
    data_path, metadata_path,
    ["Note", "Exact_L", "Len/BPM", "Micro"],
    ["Len_M", "Melodic_Charge"],
    ["Micro"],
    test_data_only=True
)


def objective(dataset, config, *, N, sr, lr, input_scaling, ridge, seed):
    trd, trt, vad, vat = dataset

    trd_keys = list(trd.keys())
    trd = [trd[k] for k in trd_keys]
    trt = [trt[k] for k in trd_keys]

    trd, trt, = map(lambda abc: [np.squeeze(one_piece.to_numpy()) for one_piece in abc], [trd, trt])

    trt = [i.reshape(-1, 1) for i in trt]

    print(trd[0].shape, trt[0].shape, vad.shape, vat.shape)

    instances = config['instances_per_trial']

    # the seed should change between each trial to prevent bias
    trial_seed = seed

    losses = []
    for i in range(instances):
        model = create_model(input_scaling, N, sr, lr, ridge, trial_seed)

        
        model.fit(trd, trt)

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
                piece_mse += (predictions[j][0] - test_goals_vel[piece].iloc[j]["Micro"])**2
                no_model_mse += (test_goals_vel[piece].iloc[j]["Micro"])**2
                #print(f'{predictions[i][0]}\t{test_goals_vel[piece].iloc[i]["Micro"]}')
            total_mse += piece_mse
            total_mse_len += piece_len
            piece_mse /= piece_len
            print(f"Piece {i}: {piece_mse}")
        print(f"All pieces: {total_mse / total_mse_len}")
        print(f"No model: {no_model_mse / total_mse_len}")

        loss = total_mse / total_mse_len
        losses.append(loss)

        trial_seed += 1

    with open(f"/stash/tlab/theom_intern/distributed_reservoir_runs/{save_name}/{cpu_name}_hp_search/{cpu_name}_all_hps.txt", "a") as f:
        f.write(f"\n{N}\t{sr}\t{lr}\t{ridge}\t{input_scaling}\t{np.mean(losses)}")
    
    return {'loss': np.mean(losses)}


if args.tune:
    best = research(objective, [trd, trt, vad, vat], f"/stash/tlab/theom_intern/distributed_reservoir_runs/{save_name}/{cpu_name}_hp_search/{cpu_name}.config.json")
    with open(f"/stash/tlab/theom_intern/distributed_reservoir_runs/{save_name}/{cpu_name}_hp_search/{cpu_name}_best_hps.txt", "a") as f:
        f.write(str(best))
    fig = plot_hyperopt_report(f"/stash/tlab/theom_intern/distributed_reservoir_runs/{save_name}", ("lr", "sr", "ridge"), metric="loss")
    fig.savefig("/stash/tlab/theom_intern/figure1.png")
    fig.show()


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

    store_model(model)
else:
    pass
    # model_to_test = load_model(name)

# test model

