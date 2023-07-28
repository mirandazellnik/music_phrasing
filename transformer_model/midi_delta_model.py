"""
https://keras.io/examples/nlp/neural_machine_translation_with_transformer/
"""

import os
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
from util.format_midi_delta_dataset import format_midi_dataset

data_paths = ["/stash/tlab/theom_intern/midi_data/asap-dataset-master/Chopin/Scherzos/20"]
save_name = sys.argv[1]



sequence_length = 100
batch_size = 64
num_samples = 40000
sequence_offset = 100

train_ds, val_ds, test_pairs, notes_vectorization, train_pairs = format_midi_dataset(data_paths, sequence_length, sequence_offset, batch_size)

vocab_size_notes = notes_vectorization.vocabulary_size()

#embed_dim = 128
#latent_dim = 8192
num_heads = 8


def custom_loss(y_true, y_pred):
    
    return K.mean(K.square(y_pred - y_true), axis=-1)-(K.mean(K.square(y_pred - .52/127*100), axis=-1)*.1)


def create_model(embed_dim, latent_dim, learning_rate):
    encoder_inputs = keras.Input(shape=(None,), dtype="int64", name="encoder_inputs")
    x = PositionalEmbedding(sequence_length*2, vocab_size_notes, embed_dim)(encoder_inputs)
    encoder_outputs = TransformerEncoder(embed_dim, latent_dim, num_heads)(x)
    encoder_outputs = layers.Dropout(.5)(encoder_outputs)
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
        opt, loss="mean_squared_error", metrics=["mean_absolute_error", "cosine_similarity"]
    )

    return transformer

def optimize_model(hp):
    embed_dim = hp.Int("embed_dim", min_value=32, max_value=128, step=32)
    latent_dim = hp.Int("latent_dim", min_value=1024, max_value=12288, step=1024)
    lr = hp.Float("lr", min_value = .00005, max_value = .005, sampling="log")


    transformer = create_model(embed_dim, latent_dim, lr)

    return transformer


def decode_sequence(transformer, input_sentence):
    tokenized_input_sentence = notes_vectorization([input_sentence])
    output_vols = [np.zeros(sequence_length*2+1)]
    for i in range(len(input_sentence.split(" "))-1):
        predictions = transformer([tokenized_input_sentence, tf.constant([output_vols[0][:-1]])])
        output_vols[0][i+1]=(round(float(predictions[0, i, 0]), 3))
    return output_vols


train = True
epochs = 400


if train:
    test_notes_texts = [pair[0] for pair in test_pairs]

    
    

    #tuner.search_space_summary()

    
    #early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4)

    tensorboard = keras.callbacks.TensorBoard(f"/stash/tlab/theom_intern/logs/{save_name}/tensorboard", profile_batch="220,260")

    checkpoint = keras.callbacks.ModelCheckpoint(f"/stash/tlab/theom_intern/logs/{save_name}/checkpoints", monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='auto', period=1)

    strategy = tf.distribute.MirroredStrategy()

    with strategy.scope():
        model = create_model(64, 12288, .0007)

        try:
            model.load_weights(f"/stash/tlab/theom_intern/logs/{save_name}/checkpoints")
            print("CONTINUING PROGRESS.")
        except:
            model = create_model(64, 12288, .0007)
            print("NO SAVE FOUND")
    
    model.fit(train_ds, epochs=epochs, validation_data=val_ds, verbose=2, callbacks = [checkpoint, tensorboard])
    

    #with strategy.scope():
    #    tuner = keras_tuner.Hyperband(
    #        hypermodel=optimize_model,
    #        objective="val_loss",
    #        max_epochs=125,
    #        executions_per_trial=1,
    #        overwrite=False,
    #        directory=f"/stash/tlab/theom_intern/logs/{save_name}/tuner",
    #        project_name="tuner",
    #    )

    #    tuner.reload()

        #tuner.search(train_ds, validation_data=val_ds, verbose=2, callbacks=[tensorboard])

    #best_hps = tuner.get_best_hyperparameters(5)
    #pickle_out = open(f"/stash/tlab/theom_intern/logs/{save_name}/best_hps", "wb")
    #pickle.dump(best_hps, pickle_out)
    #pickle_out.close()

    #best_models = tuner.get_best_models(num_models=1)
    #pickle_out = open(f"/stash/tlab/theom_intern/logs/{save_name}/best_models", "wb")
    #pickle.dump(best_models, pickle_out)
    #pickle_out.close()
    
    
    #transformer.summary()
    
    #keras.utils.plot_model(transformer, show_shapes=True, show_dtype=True, expand_nested=True, show_layer_activations=True, show_trainable=True)




    plotter = PlotLearning()

    #transformer.fit(train_ds, epochs=epochs, validation_data=val_ds, callbacks=[plotter], verbose=2)
            
    #model.save_weights(f"/stash/tlab/theom_intern/logs/{save_name}/weights")

#plotter.finish("filename2")

#model = best_models[0]

#transformer = create_model()



print("TRAIN _______________________________________")
for k in range(20):
    x = random.randint(0, len(train_pairs)-1)
    notes = train_pairs[x][0]
    expected = train_pairs[x][1]

    translated = decode_sequence(model, notes)
    translated = translated[0][:len(notes.split(" "))+1]

    final = [0]
    for i in range(int(len(translated)/2)):
        final.append(final[i]-(final[i]-round(translated[i+1]))*translated[i])


    print(f"IN: {notes}")
    print(f"EXP: {' '.join(str(x) for x in expected)}")
    print(f"OUT(o): {' '.join(str(int(x*100)) for i, x in enumerate(translated))}")
    print(f"OUT: {' '.join(str(int(x*128)) for i, x in enumerate(final) if i > 0)}")


print("TEST _______________________________________")


model = create_model(64, 12288, .0007)
#saves = [int(name[4:]) for name in os.listdir(f"/stash/tlab/theom_intern/logs/{save_name}/checkpoints/")]

model.load_weights(f"/stash/tlab/theom_intern/logs/{save_name}/checkpoints")

test_notes_texts = [pair[0] for pair in test_pairs]

for k in range(20):
    x = random.randint(0, len(test_pairs)-1)
    notes = test_pairs[x][0]
    expected = test_pairs[x][1]

    translated = decode_sequence(model, notes)
    translated = translated[0][:len(notes.split(" "))+1]

    final = [0]
    for i in range(int(len(translated)/2)):
        final.append(final[i]-(final[i]-round(translated[i+1]))*translated[i])


    print(f"IN: {notes}")
    print(f"EXP: {' '.join(str(x) for x in expected)}")
    print(f"OUT(o): {' '.join(str(int(x*100)) for i, x in enumerate(translated))}")
    print(f"OUT: {' '.join(str(int(x*128)) for i, x in enumerate(final) if i > 0)}")

#plotter.finish("filename2")