import sys
import os
import argparse

import tensorflow
from tensorflow import keras
import keras_tuner
import pickle
import numpy as np
from reservoirpy.nodes import Reservoir, Ridge, FORCE

from util.load_data import prepare_dataset

data_path = "/stash/tlab/theom_intern/midi_data/asap-dataset-processed/shifted_by_piece.json"
metadata_path = "/stash/tlab/theom_intern/midi_data/asap-dataset-master/metadata.csv"


argParser = argparse.ArgumentParser()
argParser.add_argument("name", nargs="?", help="Name of this run, for logging, model saving, etc.", type=str)
argParser.add_argument("-t", "--no-train", help="Don't train a new model, instead load the existing model.", action=argparse.BooleanOptionalAction)
argParser.add_argument("-T", "--tune", help="Tune the model with hyperband.", action=argparse.BooleanOptionalAction)
argParser.add_argument("-g", "--goal", help="Goal variable")
args = argParser.parse_args()

assert args.name
save_name = args.name
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

tensorboard = keras.callbacks.TensorBoard(f"/stash/tlab/theom_intern/ts_logs/{save_name}/tensorboard")

def storeModel(model, subname="main"):
    # Its important to use binary mode
    os.makedirs(os.path.dirname(f"/stash/tlab/theom_intern/res_models/{save_name}/{subname}/model.pkl"), exist_ok=True)
    dbfile = open(f"/stash/tlab/theom_intern/res_models/{save_name}/{subname}/model.pkl", 'ab')
    # source, destination
    pickle.dump(model, dbfile)                    
    dbfile.close()
 
def loadModel(subname="main"):
    # for reading also binary mode is important
    dbfile = open(f"/stash/tlab/theom_intern/res_models/{save_name}/{subname}/model.pkl", 'rb')    
    model = pickle.load(dbfile)
    return model

def create_model(lr=.5, sr=.9):
    
    #normalizer = keras.layers.Normalization(axis=-1)
    #normalizer.adapt(trd)

    reservoirs = [Reservoir(100, lr=lr, sr=sr) for i in range(len(np.squeeze(trd[list(trd.keys())[0]].to_numpy())[0]))]
    print(f"{len(reservoirs)} Res[]s")
    ridge = FORCE()

    esn = reservoirs >> ridge

    return esn

def optimizer(hp):

    first_layer = hp.Int("first_layer", min_value=10, max_value=2560, step=2, sampling='log')
    second_layer = hp.Int("second_layer", min_value=5, max_value=2560, step=2, sampling='log')
    dropout_1 = hp.Float("d1", min_value = 0, max_value = .5, step=.1)
    dropout_2 = hp.Float("d2", min_value = .0, max_value = .5, step=.1)
    lr = hp.Float("lr", min_value = .00001, max_value = .01, sampling="log")

    model = create_model(first_layer, second_layer, dropout_1, dropout_2, lr)

    return model

if args.tune:

    tuner = keras_tuner.Hyperband(
        hypermodel=create_model,
        objective="val_loss",
        max_epochs=1,
        executions_per_trial=5,
        overwrite=False,
        directory=f"/stash/tlab/theom_intern/ts_logs/{save_name}/tuner",
        project_name="tuner",
    )


    tuner.search(trd, trt, validation_data=(vad, vat), verbose=2, callbacks=[tensorboard])

    best_hps = tuner.get_best_hyperparameters(5)
    pickle_out = open(f"/stash/tlab/theom_intern/logs/{save_name}/best_hps", "wb")
    pickle.dump(best_hps, pickle_out)
    pickle_out.close()

    best_models = tuner.get_best_models(num_models=1)
    pickle_out = open(f"/stash/tlab/theom_intern/logs/{save_name}/best_models", "wb")
    pickle.dump(best_models, pickle_out)
    pickle_out.close()

else:

    if not args.no_train:
        models = []
        for i in range(1):
            print(f"----------------------------------- MODEL {i+1} -----------------------------------")
            model = create_model()
            #reservoir, readout = Reservoir(100, lr=0.2, sr=1.0), Ridge(ridge=1e-3)

            pieces = list(trd.keys())
            print(len(pieces))

            for i, piece in enumerate(pieces):

                train_piece_data = trd[piece]
                train_piece_goal = trt[piece]

                train_piece_data = np.squeeze(train_piece_data.to_numpy())
                train_piece_goal = np.squeeze(train_piece_goal.to_numpy())
                print("==============================")
                train_piece_goal = np.expand_dims(train_piece_goal, axis=1)
                print(train_piece_data.shape)
                print(train_piece_goal.shape)
                print("==============================")

                piece_len = train_piece_data.shape[0]
                                
                model.train(train_piece_data, train_piece_goal, reset=True)

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

        storeModel(model)

    else:
        model = loadModel()
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

            # PRINT PREDICTED, EXPECTED
            predictions = model.run(vad, reset=True)
            #for j in range(len(predictions)):
            #    print(f"{predictions[j][0]}\t{test_goals_vel[piece].iloc[j]['Micro']}")


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


    model = keras.models.load_model(f"/stash/tlab/theom_intern/models/{save_name}/{goal}")

    for row in range(0, 600):
        if goal == "Micro" and row > 0:
            ted.loc[row, "-1_Micro"] = float(out) 
        inputs = ted.loc[row]
        if row == 0:
            print(inputs)

        out = model(inputs)[0][0]
        exp = f"{tet.iloc[row]['Micro']}"
        #de = f"{ted.iloc[row]['Len_M']}"
        print(f"{exp}\t{out}")
