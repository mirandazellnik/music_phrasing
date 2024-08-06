import argparse
import json
import random
from functools import wraps

import pandas
import matplotlib.pyplot as plt
import pickle
import numpy as np

from reservoirpy.observables import mse, rsquare

from util.load_data import prepare_dataset

data_path = "/stash/tlab/theom_intern/midi_data/asap-dataset-processed/shifted_by_piece.json"
metadata_path = "/stash/tlab/theom_intern/midi_data/asap-dataset-master/metadata.csv"

argParser = argparse.ArgumentParser()
argParser.add_argument("model", nargs="?", help="Name of the model to use.", type=str)
argParser.add_argument("-g", "--goal", help="Goal variable")
argParser.add_argument("-r", "--reservoir", help="The model is a reservoir rather than a simple NN.", action=argparse.BooleanOptionalAction)

args = argParser.parse_args()

assert args.model
model_name = args.model
goal = args.goal


trd, trt, vad, vat, ted, tet = prepare_dataset(
    data_path, metadata_path,
    ["Note", "Exact_L", "Len/BPM"],
    ["Len_M", "Melodic_Charge", "Micro"],
    ["Micro"],
    train_by_piece=True,
    val_by_piece=True,
    sample_repeats=5
)
'''
print(len(vad.keys()))

total_notes = 0
for key in vad.keys():
    total_notes += vad[key].shape[0]
print(total_notes)
'''


def _elementwise(func):
    """Vectorize a function to apply it
    on arrays.
    """
    vect = np.vectorize(func)

    @wraps(func)
    def vect_wrapper(*args, **kwargs):
        u = np.asanyarray(args)
        v = vect(u)
        return v[0]

    return vect_wrapper

@_elementwise
def relu_test(x: np.ndarray) -> np.ndarray:

    if x < 0:
        return 0.0
    return x


loss = 0
r2 = 0
for piece in vad.keys():
    print(piece)

    piece_data_vel = vad[piece]
    
    piece_data_vel = np.squeeze(piece_data_vel.to_numpy())
    
    model_res = pickle.load( open( f"/stash/tlab/theom_intern/res_models/{model_name}.p", "rb" ) )
    
    vel_predictions = model_res.run(piece_data_vel, reset=True)
    
    a, b = [vel_predictions[j][0] for j in range(len(vel_predictions))], [vat[piece].iloc[j]["Micro"] for j in range(len(vel_predictions))]

    loss += mse(a, b)
    r2 += rsquare(b, a)

    if piece == list(vad.keys())[0]:
        for i in range(len(a)):
            print(f'{a[i]} {b[i]}')

loss /= len(vad.keys())
r2 /= len(vad.keys())

print('loss', loss)
print('r2', r2)
