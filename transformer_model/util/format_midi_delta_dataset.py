import os
import random
import string
import re
import numpy as np
from musicautobot.numpy_encode import PIANO_RANGE, VALTSEP
import tensorflow as tf
from tensorflow import keras
layers = keras.layers

from util.midi_to_data import get_data

def vectorize(vectorization):
    """
        Return the vectorization function that turns (notes, vols) into (inputs_vectorized, outputs)
    """
    def map_vectors(notes, vols):
        print("AAAAAAAAAAAa", notes.shape)
        notes = vectorization(notes)
        print("AAAAA", vols.shape, type(vols), notes.shape)

        return ({"encoder_inputs": notes, "decoder_inputs": vols[:, :-1],}, vols[:, 1:])
    return map_vectors

def pairs_to_ds(pairs, sequence_length, batch_size, notes_vectorization):
    """
        Turn a set of (unvectorized_notes, vols) into vectorized and correctly sized notes, and correctly sized vols)
    """
    notes_texts, vols_texts = zip(*pairs)
    notes_texts = list(notes_texts)
    vols_texts_int = list(vols_texts)

    vols_texts = []
    for ind, i in enumerate(vols_texts_int):
        temp = [0]
        past_vols = [0]

        for j, vol in enumerate(i):
            c = float(vol)/128 + (1/256)
            past_vols.append(c)
            temp.extend([(past_vols[j] - c) / (past_vols[j] - (c > past_vols[j])), .99 if c > past_vols[j] else (.01 if c < past_vols[j] else .5)])

        vols_texts.append(temp)


    fullsize_vols_texts = []
    for i in range(len(notes_texts)):
        vols = [vol for vol in vols_texts[i]]

        vols += [0 for k in range(sequence_length*2 + 1 - len(vols))]

        fullsize_vols_texts.append(vols)
    
    print(vols_texts_int[0])
    print(notes_texts[0])
    print(fullsize_vols_texts[0])

    dataset = tf.data.Dataset.from_tensor_slices((notes_texts, fullsize_vols_texts))
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(vectorize(notes_vectorization))
    return dataset.shuffle(2048).prefetch(16).cache()

def format_midi_dataset(data_paths, sequence_length, sequence_offset, batch_size):
    
    # Find all .mid files within the data_paths given, and get them in processed form
    pieces = {}
    for path in data_paths:
        if path[-4:] == ".mid": # If it's a midi file, read that one file
            pieces[path] = get_data(path)
        elif os.path.isdir(path): # If it's a directory, traverse all subfolders for all midi files
            for (root, dirs, filenames) in os.walk(path):
                for f in filenames:
                    if f[-4:] == ".mid":
                        pieces[os.path.join(root, f)] = get_data(os.path.join(root, f))
        else:
            print(f"COULD NOT FIND MIDI FILE(s) AT: {path}")

    # Turn the (notes, vols) arrays into correctly formatted inputs+outputs, and shift the (notes) into all 12 keys
    def shift_all_12(notes, vols, i):
        return [
            (
                " ".join(map(lambda x: f"n{min(PIANO_RANGE[1], max(PIANO_RANGE[0], x[0]+shift)) if x[0] != VALTSEP else '00'} d{x[1]}", notes[i*sequence_offset:i*sequence_offset+sequence_length])),
                vols[i*sequence_offset:i*sequence_offset+sequence_length]
            ) for shift in range(-5, 7)
        ]

    # Split the dict of full unprocessed pieces into sequence_length sized chunks, and turn the [notes, lengths] into strings: e.g. "piecename":[["notes", [vols]], ...]
    pieces_pairs = {}
    for path, (notes, vols) in pieces.items():
        piece = []

        for i in range(int((len(notes)-sequence_length*2)/sequence_offset)+1):
            piece.extend(shift_all_12(notes, vols, i))
        #if ((len(notes)-sequence_length*2) % sequence_offset) > 0:
        #    piece.extend(shift_all_12(notes, vols, 0))
        pieces_pairs[path] = piece
    
    pieces_names = list(pieces_pairs.keys())
    pieces_names.sort()
    random.Random(1).shuffle(pieces_names)

    num_val_pieces = int(0.15 * len(pieces_names))
    num_train_pieces = len(pieces_names) - 2 * num_val_pieces
    train_pieces = pieces_names[:num_train_pieces]
    val_pieces = pieces_names[num_train_pieces : num_train_pieces + num_val_pieces]
    test_pieces = pieces_names[num_train_pieces + num_val_pieces :]

    # Create a flat list of input/output pairs for each group (training, val, test)
    train_pairs = [a for b in [pieces_pairs[name] for name in train_pieces] for a in b]
    val_pairs = [a for b in [pieces_pairs[name] for name in val_pieces] for a in b]
    test_pairs = [a for b in [pieces_pairs[name] for name in test_pieces] for a in b]
    
    random.shuffle(train_pairs)
    random.shuffle(val_pairs)
    random.shuffle(test_pairs)

    print(train_pairs[0])

    print(f"Total: {len(pieces_names)} pieces = {len(train_pairs) + len(val_pairs) + len(test_pairs)} pairs")
    print(f"Training: {len(train_pieces)} pieces = {len(train_pairs)} pairs")
    print(f"Validation: {len(val_pieces)} pieces = {len(val_pairs)} pairs")
    print(f"Testing: {len(test_pieces)} pieces = {len(test_pairs)} pairs")


    """c = [pair[0] for pair in train_pairs + val_pairs + test_pairs]
    
    count = {}
    for i in c:
        for a in i.split(' '):
            if a[0] == "n":
                continue
            if a in count:
                count[a] += 1
            else:
                count[a] = 1
    
    for i in count:
        l = []
        for j in c:
            x = j.split(' ')
            if 'n00' in x[0::2]:
                break
        else:
            print(i)"""

    max_note_len = 64

    vocab = ["n00"] + [f"n{x}" for x in range(PIANO_RANGE[0], PIANO_RANGE[1]+1)] + [f"d{x}" for x in [1,2,3,4,6,8,10,12,14,16,20,24,28,32]]

    notes_vectorization = layers.TextVectorization(
        max_tokens=None, output_mode="int", output_sequence_length=sequence_length*2, vocabulary=vocab
    )

    #train_notes_texts = [pair[0] for pair in train_pairs]
    #notes_vectorization.adapt([pair[0] for pair in train_pairs + val_pairs + test_pairs])
    #print(notes_vectorization.get_vocabulary())

    #return notes_vectorization

    train_ds = pairs_to_ds(train_pairs, sequence_length, batch_size, notes_vectorization)
    val_ds = pairs_to_ds(val_pairs, sequence_length, batch_size, notes_vectorization)

    return train_ds, val_ds, test_pairs, notes_vectorization, train_pairs