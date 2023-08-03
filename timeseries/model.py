import sys

import tensorflow
from tensorflow import keras
import keras_tuner
import pickle
import numpy as np

from util.load_data import prepare_dataset

data_path = "/stash/tlab/theom_intern/midi_data/asap-dataset-processed/shifted_by_piece.json"
metadata_path = "/stash/tlab/theom_intern/midi_data/asap-dataset-master/metadata.csv"

save_name = None
if len(sys.argv) > 1:
    save_name = sys.argv[1]

# Micro data (current and -1): Note, Exact_Lower, Exact_Higher, Motion
# Micro data (current only): Len_M, W.5, B.1, B.5, A.1, A.5, W.5
# Macro data: Time, B1, B2, B4, W.5

print("Preparing dataset...")
trd, trt, vad, vat, ted, tet = prepare_dataset(data_path, metadata_path, ["Len_M", "D_A", "Len/BPM", "Len_Ratio"], ["Exact", "Len_P", "B.50", "B2.0", "A.50", "A2.0", "W.50", "W2.0"], ["Len_P"])

tensorboard = keras.callbacks.TensorBoard(f"/stash/tlab/theom_intern/ts_logs/{save_name}/tensorboard")

def create_model(first_layer, second_layer, dropout_1, dropout_2, lr):
    
    normalizer = keras.layers.Normalization(axis=-1)
    normalizer.adapt(trd)

    model = keras.models.Sequential()

    model.add(normalizer)
    model.add(keras.layers.Dense(first_layer, activation='relu'))
    model.add(keras.layers.Dropout(dropout_1))
    model.add(keras.layers.Dense(second_layer, activation='relu'))
    model.add(keras.layers.Dropout(dropout_2))
    model.add(keras.layers.Dense(2, activation='relu'))

    model.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=lr), metrics=None)

    return model

def optimizer(hp):

    first_layer = hp.Int("first_layer", min_value=10, max_value=2560, step=2, sampling='log')
    second_layer = hp.Int("second_layer", min_value=5, max_value=2560, step=2, sampling='log')
    dropout_1 = hp.Float("d1", min_value = 0, max_value = .5, step=.1)
    dropout_2 = hp.Float("d2", min_value = .0, max_value = .5, step=.1)
    lr = hp.Float("lr", min_value = .00001, max_value = .01, sampling="log")

    model = create_model(first_layer, second_layer, dropout_1, dropout_2, lr)

    return model

"""
tuner = keras_tuner.Hyperband(
    hypermodel=create_model,
    objective="val_loss",
    max_epochs=50,
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
"""

model = create_model(80, 80, .2, .2, 4.8276e-5)
model.fit(trd, trt, validation_data=(vad, vat), epochs=3, batch_size=32, verbose=2, callbacks=[tensorboard] if save_name else [])


for row in range(300):
    #old_micro = float(out) if row > 0 else 0.0
    #ted.loc[row, "-1_Micro"] = old_micro 
    inputs = ted.loc[row]

    out = f"{model(inputs)[0][0]}"
    exp = f"{tet.iloc[row]['Len_P']}"
    de = f"{ted.iloc[row]['Len_M']}"
    print(f"{exp}\t{out}\t{de}")
    out = model(inputs)[0][0]
print("----------")
for row in range(300):
    #old_micro = float(out) if row > 0 else 0.0
    #ted.loc[row, "-1_Micro"] = old_micro 
    inputs = trd.loc[row]

    out = f"{model(inputs)[0][0]}"
    exp = f"{trt.iloc[row]['Len_P']}"
    de = f"{trd.iloc[row]['Len_M']}"
    print(f"{exp}\t{out}\t{de}")
    out = model(inputs)[0][0]
