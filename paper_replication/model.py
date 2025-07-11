import sys
import os
import argparse

import tensorflow
from tensorflow import keras
import keras_tuner
import pickle
import numpy as np
from keras import backend as K

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
    trd, trt, ted, tet, vad, vat = prepare_dataset(
        data_path, metadata_path,
        ["Note", "Exact_L", "Len/BPM"],
        ["Len_M", "Melodic_Charge", "Micro"],
        ["Micro"],
    )
    """
    trd, trt, vad, vat, ted, tet = prepare_dataset(
        data_path, metadata_path,
        ["Note", "Exact_L", "Len/BPM", "Micro"],
        ["Len_M", "Melodic_Charge"],
        ["Micro"]
    )"""
elif goal == "Len_P":
    trd, trt, vad, vat, ted, tet = prepare_dataset(
        data_path, metadata_path,
        ["Len_M", "D_A", "Len/BPM", "Len_Ratio"],
        ["Exact", "Len_P", "Micro", "B.50", "B2.0", "A.50", "A2.0", "W.50", "W2.0"],
        ["Len_P"]
    )

tensorboard = keras.callbacks.TensorBoard(f"/stash/tlab/theom_intern/ts_logs/{save_name}/tensorboard")

def create_model(first_layer, second_layer, dropout_1, dropout_2, lr):
    
    normalizer = keras.layers.Normalization(axis=-1)
    normalizer.adapt(trd.to_numpy())


    model = keras.models.Sequential()

    model.add(normalizer)
    model.add(keras.layers.Dense(first_layer, activation='relu'))
    #model.add(keras.layers.Dropout(dropout_1))
    #if second_layer:
    #    model.add(keras.layers.Dense(second_layer, activation='relu'))
    #    model.add(keras.layers.Dropout(dropout_2))
    model.add(keras.layers.Dense(1, activation='linear'))



    model.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=lr), metrics=None)

    return model

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
            model = create_model(80, 80, .2, .2, 4.8276e-5)
            model.fit(trd, trt, validation_data=(vad, vat), epochs=5, batch_size=32, verbose=2, callbacks=[tensorboard] if save_name else [])
            models.append(model)
        os.makedirs(f"/stash/tlab/theom_intern/models/{save_name}/{goal}")
        model.save(f"/stash/tlab/theom_intern/models/{save_name}/{goal}/model.keras")

    model = keras.models.load_model(f"/stash/tlab/theom_intern/models/{save_name}/{goal}/model.keras")

    loss = 0
    loss_nomodel = 0
    outs = []
    exps = []

    for row in range(0, 24466):
        #if goal == "Micro" and row > 0:
        #    ted.loc[row, "-1_Micro"] = float(out) 
        inputs = ted.loc[row].to_numpy()
        inputs = inputs[np.newaxis, :]
        
        if row == 0:
            print(inputs)

        out = model(inputs)[0][0]
        exp = f"{tet.iloc[row]['Micro']}"
        #de = f"{ted.iloc[row]['Len_M']}"
        #print(f"{exp}\t{out}")
        loss += (float(exp) - float(out))**2
        loss_nomodel += (float(exp))**2
        outs.append(float(out))
        exps.append(float(exp))

    loss /= 24466
    loss_nomodel /= 24466
    print(f"MSE: {loss}")
    print(f"MSE(no model): {loss_nomodel}")

    def coeff_determination(y_true, y_pred):
        SS_res =  sum([(y_true[i]-y_pred[i])**2 for i in range(len(y_true))])
        SS_tot =  sum([(y_true[i]-(sum(y_true)/len(y_true)))**2 for i in range(len(y_true))])
        return ( 1 - SS_res/(SS_tot) )
    
    r2 = coeff_determination(exps, outs)
    print(f"r2: {r2}")

    r2_no = coeff_determination(exps, [0 for i in range(len(exps))])
    print(f"r2_no: {r2_no}")
    

