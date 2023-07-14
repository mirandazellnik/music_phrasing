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
    
    combined_notes_texts = []
    combined_vols_texts = []
    for i in range(len(pairs)):
        vols = [0] + [vol for vol in vols_texts[i] if vol > 0]

        vols += [0 for k in range(sequence_length + 1 - len(vols))]

        combined_notes_texts.append(notes_texts[i])
        combined_vols_texts.append(vols)
            

    dataset = tf.data.Dataset.from_tensor_slices((combined_notes_texts, combined_vols_texts))
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(vectorize(notes_vectorization))
    return dataset.shuffle(2048).prefetch(16).cache()

def format_midi_dataset(data_paths, sequence_length, sequence_offset, batch_size):
    print("MIDI FORMATTING")
    for path in data_paths:
        notes, vols = data_from_midi(path)
        text_pairs = []
        for i in range(int((len(notes)-sequence_length)/sequence_offset)):
            text_pairs.append((" ".join(map(str, notes[i*sequence_offset:i*sequence_offset+sequence_length])), vols[i*sequence_offset:i*sequence_offset+sequence_length]))

    random.shuffle(text_pairs)
    num_val_samples = int(0.15 * len(text_pairs))
    num_train_samples = len(text_pairs) - 2 * num_val_samples
    train_pairs = text_pairs[:num_train_samples]
    val_pairs = text_pairs[num_train_samples : num_train_samples + num_val_samples]
    test_pairs = text_pairs[num_train_samples + num_val_samples :]

    print(f"{len(text_pairs)} total pairs")
    print(f"{len(train_pairs)} training pairs")
    print(f"{len(val_pairs)} validation pairs")
    print(f"{len(test_pairs)} test pairs")

    notes_vectorization = layers.TextVectorization(
        max_tokens=128, output_mode="int", output_sequence_length=sequence_length,
    )

    train_notes_texts = [pair[0] for pair in train_pairs]
    notes_vectorization.adapt(train_notes_texts)

    train_ds = pairs_to_ds(train_pairs, sequence_length, batch_size, notes_vectorization)
    val_ds = pairs_to_ds(val_pairs, sequence_length, batch_size, notes_vectorization)
    return train_ds, val_ds, test_pairs, notes_vectorization
