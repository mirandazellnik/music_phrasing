import sys
import os
import argparse

import tensorflow
from tensorflow import keras
import keras_tuner
import pickle
import numpy as np

from util.load_data import prepare_dataset
from util.generate_midi import apply_outputs

data_path = "/stash/tlab/theom_intern/midi_data/asap-dataset-processed/shifted_by_piece.json"
metadata_path = "/stash/tlab/theom_intern/midi_data/asap-dataset-master/metadata.csv"

ASAP_PATH = "/stash/tlab/theom_intern/midi_data/asap-dataset-master"
PROCESSED_PATH = "/stash/tlab/theom_intern/midi_data/asap-dataset-processed"


argParser = argparse.ArgumentParser()
argParser.add_argument("model", nargs="?", help="Name of the model to use.", type=str)
argParser.add_argument("-g", "--goal", help="Goal variable")
args = argParser.parse_args()

assert args.model
model_name = args.model
goal = args.goal
if not goal:
    goal = "Micro"
assert goal in ["Micro", "Len_P"]


test_data_vel, test_goals_vel = prepare_dataset(
    data_path, metadata_path,
    ["Note", "Exact_L", "Exact_H", "Motion", "Micro"],
    ["Len_M", "W.50", "B.10", "B.50", "A.10", "A.50", "W.50"],
    ["Micro"],
    test_data_only=True
)

test_data_len, test_goals_len = prepare_dataset(
    data_path, metadata_path,
    ["Len_M", "D_A", "Len/BPM", "Len_Ratio"],
    ["Exact", "Len_P", "Micro", "B.50", "B2.0", "A.50", "A2.0", "W.50", "W2.0"],
    ["Len_P"],
    test_data_only=True
)

piece = list(test_data_vel.keys())[8]

piece_data_vel = test_data_vel[piece]
piece_data_len = test_data_len[piece]

model_vel = keras.models.load_model(f"/stash/tlab/theom_intern/models/{model_name}/{goal}")
model_len = keras.models.load_model(f"/stash/tlab/theom_intern/models/{model_name}/Len_P")


outputs_vel = []
outputs_len = []

for row in range(len(piece_data_vel)):

    if goal == "Micro" and row > 0:
        piece_data_vel.loc[row, "-1_Micro"] = float(out) 
    inputs = piece_data_vel.loc[row]
    out = model_vel(inputs)[0][0]
    outputs_vel.append(out)

    exp = f"{test_goals_vel[piece].iloc[row]['Micro']}"

    piece_data_len.loc[row, "Micro"] = float(out) 
    inputs2 = piece_data_len.loc[row]
    out_len = model_len(inputs2)[0][0]
    outputs_len.append(out_len)

    #de = f"{ted.iloc[row]['Len_M']}"
    print(f"{exp}\t{out}")


perf_path = os.path.join(PROCESSED_PATH, piece)[:-4] + ".txt"
apply_outputs(perf_path, "/stash/tlab/theom_intern/midi_data/output_len.mid", outputs_vel, outputs_len)