"""
https://keras.io/examples/nlp/neural_machine_translation_with_transformer/
"""

import random
import string
import re
import numpy as np
import tensorflow as tf
from tensorflow import keras
import keras_tuner
import pickle
import sys

layers, K = keras.layers, keras.backend

from util.plotter import PlotLearning
from util.layers import TransformerEncoder, TransformerDecoder, PositionalEmbedding
from util.format_midi_dataset import format_midi_dataset

data_path = "/stash/tlab/theom_intern/midi_data/little_pieces_4_(c)oguri.mid"
save_name = sys.argv[1]


vocab_size_notes = 62
sequence_length = 200
batch_size = 128
num_samples = 40000
sequence_offset = 10

train_ds, val_ds, test_pairs, notes_vectorization = format_midi_dataset([data_path], sequence_length, sequence_offset, batch_size)

#embed_dim = 128
#latent_dim = 8192
num_heads = 8



def create_model(embed_dim, latent_dim, learning_rate):
    encoder_inputs = keras.Input(shape=(None,), dtype="int64", name="encoder_inputs")
    x = PositionalEmbedding(sequence_length, vocab_size_notes, embed_dim)(encoder_inputs)
    encoder_outputs = TransformerEncoder(embed_dim, latent_dim, num_heads)(x)
    encoder = keras.Model(encoder_inputs, encoder_outputs)

    decoder_inputs = keras.Input(shape=(None,), dtype="int64", name="decoder_inputs")
    encoded_seq_inputs = keras.Input(shape=(None, embed_dim), name="decoder_state_inputs")
    x = PositionalEmbedding(sequence_length, 1, embed_dim)(decoder_inputs)
    x = TransformerDecoder(embed_dim, latent_dim, num_heads)(x, encoded_seq_inputs)
    x = layers.Dropout(.5)(x)
    decoder_outputs = layers.Dense(1, activation="sigmoid")(x)
    x2 = layers.Reshape((-1, 1))(decoder_outputs)
    decoder = keras.Model([decoder_inputs, encoded_seq_inputs], x2)
    decoder_outputs = decoder([decoder_inputs, encoder_outputs])
    #decoder_outputs = layers.Dense(20, activation="sigmoid")(decoder_outputs)
    transformer = keras.Model(
        [encoder_inputs, decoder_inputs], decoder_outputs, name="transformer"
    )

    opt = keras.optimizers.experimental.AdamW(learning_rate=learning_rate)
    opt = tf.keras.mixed_precision.LossScaleOptimizer(opt)

    
    transformer.compile(
        opt, loss="mean_squared_error", metrics=["cosine_similarity"]
    )

    return transformer

def optimize_model(hp):
    embed_dim = hp.Int("embed_dim", min_value=192, max_value=193, step=1)
    latent_dim = hp.Int("latent_dim", min_value=2048, max_value=2049, step=1)
    lr = hp.Float("lr", min_value = .000661, max_value = .000662, sampling="log")

    transformer = create_model(embed_dim, latent_dim, lr)

    return transformer


def decode_sequence(transformer, input_sentence):
    tokenized_input_sentence = notes_vectorization([input_sentence])
    output_vols = [np.zeros(sequence_length+1)]
    for i in range(len(input_sentence.split(" "))):
        predictions = transformer([tokenized_input_sentence, tf.constant([output_vols[0][:-1]])])
        output_vols[0][i+1]=(round(float(predictions[0, i, 0]), 2))
    return output_vols


train = True
epochs = 20


if train:
    test_notes_texts = [pair[0] for pair in test_pairs]

    """
    tuner = keras_tuner.Hyperband(
        hypermodel=optimize_model,
        objective="val_loss",
        max_epochs=300,
        executions_per_trial=3,
        overwrite=True,
        directory=f"/stash/tlab/theom_intern/logs/{save_name}/tuner",
        project_name="tuner",
    )

    tuner.search_space_summary()

    """

    early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=30)

    tensorboard = keras.callbacks.TensorBoard(f"/stash/tlab/theom_intern/logs/{save_name}/tensorboard", profile_batch="220,260")

    model = create_model(64, 10240, .00024587)

    model.fit(train_ds, validation_data=val_ds, epochs=20, verbose=2, callbacks=[tensorboard, early_stop])

    """best_hps = tuner.get_best_hyperparameters(5)
    pickle_out = open(f"/stash/tlab/theom_intern/logs/{save_name}/best_hps", "wb")
    pickle.dump(best_hps, pickle_out)
    pickle_out.close()

    best_models = tuner.get_best_models(num_models=5)
    pickle_out = open(f"/stash/tlab/theom_intern/logs/{save_name}/best_models", "wb")
    pickle.dump(best_models, pickle_out)
    pickle_out.close()"""

    #transformer.summary()
    
    #keras.utils.plot_model(transformer, show_shapes=True, show_dtype=True, expand_nested=True, show_layer_activations=True, show_trainable=True)

    def custom_loss(y_true, y_pred):

        x = tf.experimental.numpy.diff(y_pred, axis=-1)
        y = tf.experimental.numpy.diff(y_true, axis=-1)
        z = K.mean(K.square(x - y))

        #print(z + K.mean(K.square(y_pred - y_true), axis=-1))
        
        return z + K.mean(K.square(y_pred - y_true), axis=-1)


    plotter = PlotLearning()

    #transformer.fit(train_ds, epochs=epochs, validation_data=val_ds, callbacks=[plotter], verbose=2)
            
    model.save_weights(f"/stash/tlab/theom_intern/logs/{save_name}/weights")

#plotter.finish("filename2")

model = create_model(64, 10240, .00024587)

#transformer = create_model()
model.load_weights(f"/stash/tlab/theom_intern/logs/{save_name}/weights")

test_notes_texts = [pair[0] for pair in test_pairs]

notes = ""
for j in range(random.randint(1,10)):
    x = random.randint(0, len(test_pairs)-1)
    notes += test_pairs[x][0] + " "
notes = notes[:-1]

for k in range(10):
    notes = " ".join([str(int(note)+1) for note in notes.split(" ")])
    translated = decode_sequence(model, notes)
    translated = translated[0][:len(notes.split(" "))+1]
    print(f"IN:  {notes}")
    print(f"OUT: {' '.join(str(int(x*100)) for i, x in enumerate(translated) if i > 0)}")

notes = ""
for j in range(random.randint(1,10)):
    x = random.randint(0, len(test_pairs)-1)
    notes += test_pairs[x][0] + " "
notes = notes[:-1]

for k in range(10):
    notes = " ".join([str(int(note)+1) for note in notes.split(" ")])
    translated = decode_sequence(model, notes)
    translated = translated[0][:len(notes.split(" "))+1]
    print(f"IN:  {notes}")
    print(f"OUT: {' '.join(str(int(x*100)) for i, x in enumerate(translated) if i > 0)}")

notes = ""
for j in range(random.randint(1,10)):
    x = random.randint(0, len(test_pairs)-1)
    notes += test_pairs[x][0] + " "
notes = notes[:-1]

for k in range(10):
    notes = " ".join([str(int(note)+1) for note in notes.split(" ")])
    translated = decode_sequence(model, notes)
    translated = translated[0][:len(notes.split(" "))+1]
    print(f"IN:  {notes}")
    print(f"OUT: {' '.join(str(int(x*100)) for i, x in enumerate(translated) if i > 0)}")

#plotter.finish("filename2")