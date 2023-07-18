import os
import random
import string
import re
import numpy as np
import tensorflow as tf
from tensorflow import keras
layers = keras.layers

from util.midi_to_data import data_from_midi

def vectorize(vectorization):
    def map_vectors(notes, vols):
        notes = vectorization(notes)
        return ({"encoder_inputs": notes, "decoder_inputs": vols[:, :-1],}, vols[:, 1:])
    return map_vectors

def pairs_to_ds(pairs, sequence_length, batch_size, notes_vectorization):
    notes_texts, vols_texts = zip(*pairs)
    notes_texts = list(notes_texts)
    vols_texts1 = list(vols_texts)

    vols_texts = []
    for i in vols_texts1:
        temp = np.zeros(201)
        
        for j, vol in enumerate(i):
            temp[j] = float(vol)/128 + (1/256)

        vols_texts.append(temp)
    

    vols_texts_new = []
    notes_texts_new = []
    for i in range(len(notes_texts)):
        for j in range(-5, 6):
            notes_texts_new.append(" ".join([str(min(126, max(1, int(x) + j))) for x in notes_texts[i].split(" ")]))
            vols_texts_new.append(vols_texts[i])
    combined_notes_texts = []
    combined_vols_texts = []
    for i in range(len(notes_texts_new)):
        vols = [0] + [vol for vol in vols_texts_new[i] if vol > 0]

        vols += [0 for k in range(sequence_length + 1 - len(vols))]

        combined_notes_texts.append(notes_texts_new[i])
        combined_vols_texts.append(vols)
            

    dataset = tf.data.Dataset.from_tensor_slices((combined_notes_texts, combined_vols_texts))
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(vectorize(notes_vectorization))
    return dataset.shuffle(2048).prefetch(16).cache()

def format_midi_dataset(data_paths, sequence_length, sequence_offset, batch_size):
    print("MIDI FORMATTING")
    text_pairs = []

    for path in data_paths:
        x = []
        if path[-4:] == ".mid":
            notes, vols = data_from_midi(path)
            for i in range(int((len(notes)-sequence_length)/sequence_offset)):
                x.append((" ".join(map(str, notes[i*sequence_offset:i*sequence_offset+sequence_length])), vols[i*sequence_offset:i*sequence_offset+sequence_length]))
        else:
            for (root, dirs, filenames) in os.walk(path):
                for f in filenames:
                    x = []
                    notes, vols = data_from_midi(root + "/" + f)
                    for i in range(int((len(notes)-sequence_length)/sequence_offset)):
                        x.append((" ".join(map(str, notes[i*sequence_offset:i*sequence_offset+sequence_length])), vols[i*sequence_offset:i*sequence_offset+sequence_length]))
                    text_pairs.append(x)

    random.shuffle(text_pairs)
    text_pairs = [a for b in text_pairs for a in b]
    num_val_samples = int(0.499 * len(text_pairs))
    num_train_samples = len(text_pairs) - 2 * num_val_samples
    train_pairs = text_pairs[:num_train_samples]
    val_pairs = text_pairs[num_train_samples : num_train_samples + num_val_samples]
    test_pairs = text_pairs[num_train_samples + num_val_samples :]

    random.shuffle(train_pairs)
    random.shuffle(val_pairs)
    random.shuffle(test_pairs)

    print(f"{len(text_pairs)*12} total pairs")
    print(f"{len(train_pairs)*12} training pairs")
    print(f"{len(val_pairs)*12} validation pairs")
    print(f"{len(test_pairs)*12} test pairs")

    notes_vectorization = layers.TextVectorization(
        max_tokens=128, output_mode="int", output_sequence_length=sequence_length,
    )

    train_notes_texts = [pair[0] for pair in train_pairs]
    notes_vectorization.adapt(train_notes_texts)

    train_ds = pairs_to_ds(train_pairs, sequence_length, batch_size, notes_vectorization)
    val_ds = pairs_to_ds(val_pairs, sequence_length, batch_size, notes_vectorization)
    return train_ds, val_ds, test_pairs, notes_vectorization
