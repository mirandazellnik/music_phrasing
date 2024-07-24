import sys
import os
import argparse

import pandas
import matplotlib.pyplot as plt
import seaborn as sn
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
argParser.add_argument("-r", "--reservoir", help="The model is a reservoir rather than a simple NN.", action=argparse.BooleanOptionalAction)

args = argParser.parse_args()

assert args.model
model_name = args.model
goal = args.goal
if not goal:
    goal = "Micro"
assert goal in ["Micro", "Len_P"]


"""test_data_vel, test_goals_vel = prepare_dataset(
    data_path, metadata_path,
    ["Note", "Exact_L", "Len/BPM", "Exact_H",  "Motion", "Micro"],
    ["Len_M", "Melodic_Charge", "W.50", "B.10", "B.50", "A.10", "A.50", "W.50"],
    ["Micro"],
    test_data_only=True
)"""

test_data_len, test_goals_len = prepare_dataset(
    data_path, metadata_path,
    ["Len_M", "D_A", "Len/BPM", "Len_Ratio"],
    ["Exact", "Len_P", "Micro", "B.50", "B2.0", "A.50", "A2.0", "W.50", "W2.0"],
    ["Len_P"],
    test_data_only=True
)


test_data_vel, test_goals_vel = prepare_dataset(
    data_path, metadata_path,
    ["Note", "Exact_L", "Len/BPM"],
    ["Len_M", "Melodic_Charge", "Micro"],
    ["Micro"],
    test_data_only=True
)

os.mkdir(f"/stash/tlab/theom_intern/midi_data/{model_name}2human")
for piece in test_data_vel.keys():
    print(piece)

    piece_data_vel = test_data_vel[piece]
    piece_data_len = test_data_len[piece]

    #model_vel = keras.models.load_model(f"/stash/tlab/theom_intern/models/{model_name}/{goal}")
    model_res = pickle.load( open( f"/stash/tlab/theom_intern/res_models/{model_name}.p", "rb" ) )
    #model_len = keras.models.load_model(f"/stash/tlab/theom_intern/models/{model_name}/Len_P")


    outputs_vel = []
    outputs_len = []

    inputs_dict = piece_data_vel.to_dict(orient="list")


    vad = np.squeeze(piece_data_vel.to_numpy())
    #vad = piece_data_vel

    #outputs_vel = [o[0] for o in model_res.run(vad, reset=True)]

    for row in range(len(piece_data_vel)):

        exp = f"{test_goals_vel[piece].iloc[row]['Micro']}"

        outputs_vel.append(test_goals_vel[piece].iloc[row]['Micro'])
        #print(exp[:100])

        outputs_len.append(inputs_dict["Len_M"][row])
        
        """
        piece_data_len.loc[row, "Micro"] = float(out) 
        inputs2 = piece_data_len.loc[row]
        out_len = model_len(inputs2)[0][0]
        outputs_len.append(out_len)
        """
        #de = f"{ted.iloc[row]['Len_M']}"
        #print(f"{exp}\t{predictions[row][0]}")

    inputs_dict["output"] = outputs_vel

    #df = pandas.DataFrame(inputs_dict)

    #corr_matrix = df.corr()
    #sn.heatmap(corr_matrix, annot=True, xticklabels=1, yticklabels=1)
    #plt.show()

    print(len(outputs_vel), len(outputs_len))

    perf_path = os.path.join(PROCESSED_PATH, piece)[:-4] + ".txt"
    
    apply_outputs(perf_path, f"/stash/tlab/theom_intern/midi_data/{model_name}2human/{piece.replace('/','')}.mid", outputs_vel, outputs_len)
