import random
import string
import re
import numpy as np
import tensorflow as tf
from tensorflow import keras
layers = keras.layers

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
        temp = np.zeros(21)
        
        for j, vol in enumerate(i.split(" ")):
            if vol in ["[start]", "[end]"]:
                continue
            #temp[j] = min(.99, max(.01, float(vol)/10-.05+((random.random()-.5))))
            temp[j] = float(vol)/10 - .05
        
        vols_texts.append(temp)
    
    combined_notes_texts = []
    combined_vols_texts = []
    for i in range(len(pairs)):
        notes = ""
        vols = [0]
        for j in range(random.randint(1, 10)):
            x = random.randint(0, len(pairs)-1)
            notes += notes_texts[x] + " "
            vols += [k for k in vols_texts[x] if k > 0]
        notes = notes[:-1]
        vols += [0 for k in range(sequence_length + 1 - len(vols))]
        combined_notes_texts.append(notes)
        combined_vols_texts.append(vols)
            

    dataset = tf.data.Dataset.from_tensor_slices((combined_notes_texts, combined_vols_texts))
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(vectorize(notes_vectorization))
    return dataset.shuffle(2048).prefetch(16).cache()

def format_dataset(data_path, vocab_size_notes, sequence_length, num_samples, batch_size):
    with open(data_path) as f:
        lines = f.read().split("\n")[:-1]
    text_pairs = []
    for line in lines:
        notes, vols = line.split("\t")
        vols = "[start] " + vols + " [end]"
        text_pairs.append((notes, vols))

    text_pairs = text_pairs[:num_samples]

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

    strip_chars = string.punctuation + "¿"
    strip_chars = strip_chars.replace("[", "")
    strip_chars = strip_chars.replace("]", "")

    notes_vectorization = layers.TextVectorization(
        max_tokens=vocab_size_notes, output_mode="int", output_sequence_length=sequence_length,
    )

    train_notes_texts = [pair[0] for pair in train_pairs]
    notes_vectorization.adapt(train_notes_texts)

    train_ds = pairs_to_ds(train_pairs, sequence_length, batch_size, notes_vectorization)
    val_ds = pairs_to_ds(val_pairs, sequence_length, batch_size, notes_vectorization)
    return train_ds, val_ds, test_pairs, notes_vectorization
