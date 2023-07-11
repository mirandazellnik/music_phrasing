"""
https://keras.io/examples/nlp/neural_machine_translation_with_transformer/
"""

import random
import string
import re
import numpy as np
import tensorflow as tf
from tensorflow import keras
layers, K = keras.layers, keras.backend

from util.plotter import PlotLearning

data_path = "./data/notes_numbers.txt"

with open(data_path) as f:
    lines = f.read().split("\n")[:-1]
text_pairs = []
for line in lines:
    notes, vols = line.split("\t")
    vols = "[start] " + vols + " [end]"
    text_pairs.append((notes, vols))

text_pairs = text_pairs[:20]

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

vocab_size_notes = 62
sequence_length = 200
batch_size = 64


def custom_standardization(input_string):
    lowercase = tf.strings.lower(input_string)
    return tf.strings.regex_replace(lowercase, "[%s]" % re.escape(strip_chars), "")


notes_vectorization = layers.TextVectorization(
    max_tokens=vocab_size_notes, output_mode="int", output_sequence_length=sequence_length,
)

train_notes_texts = [pair[0] for pair in train_pairs]
notes_vectorization.adapt(train_notes_texts)

def format_dataset(notes, vols):
    notes = notes_vectorization(notes)
    return ({"encoder_inputs": notes, "decoder_inputs": vols[:, :-1],}, vols[:, 1:])


def make_dataset(pairs):
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
        for j in range(random.randint(1,3)):
            x = random.randint(0, len(pairs)-1)
            notes += notes_texts[x] + " "
            vols += [k for k in vols_texts[x] if k > 0]
        notes = notes[:-1]
        vols += [0 for k in range(201-len(vols))]
        combined_notes_texts.append(notes)
        combined_vols_texts.append(vols)
            

    dataset = tf.data.Dataset.from_tensor_slices((combined_notes_texts, combined_vols_texts))
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(format_dataset)
    return dataset.shuffle(2048).prefetch(16).cache()


train_ds = make_dataset(train_pairs)
val_ds = make_dataset(val_pairs)


class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.dense_proj = keras.Sequential(
            [layers.Dense(dense_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.supports_masking = True

    def call(self, inputs, mask=None):
        if mask is not None:
            padding_mask = tf.cast(mask[:, tf.newaxis, :], dtype="int32")
        attention_output = self.attention(
            query=inputs, value=inputs, key=inputs, attention_mask=padding_mask
        )
        proj_input = self.layernorm_1(inputs + attention_output)
        proj_output = self.dense_proj(proj_input)
        return self.layernorm_2(proj_input + proj_output)


class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, vocab_size, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.token_embeddings = layers.Embedding(
            input_dim=vocab_size, output_dim=embed_dim
        )
        self.position_embeddings = layers.Embedding(
            input_dim=sequence_length, output_dim=embed_dim
        )
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

    def call(self, inputs):
        length = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_tokens = self.token_embeddings(inputs)
        embedded_positions = self.position_embeddings(positions)
        return embedded_tokens + embedded_positions

    def compute_mask(self, inputs, mask=None):
        return tf.math.not_equal(inputs, 0)


class TransformerDecoder(layers.Layer):
    def __init__(self, embed_dim, latent_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.attention_1 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.attention_2 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.dense_proj = keras.Sequential(
            [layers.Dense(latent_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.layernorm_3 = layers.LayerNormalization()
        self.supports_masking = True

    def call(self, inputs, encoder_outputs, mask=None):
        causal_mask = self.get_causal_attention_mask(inputs)
        if mask is not None:
            padding_mask = tf.cast(mask[:, tf.newaxis, :], dtype="int32")
            padding_mask = tf.minimum(padding_mask, causal_mask)

        attention_output_1 = self.attention_1(
            query=inputs, value=inputs, key=inputs, attention_mask=causal_mask
        )
        out_1 = self.layernorm_1(inputs + attention_output_1)

        attention_output_2 = self.attention_2(
            query=out_1,
            value=encoder_outputs,
            key=encoder_outputs,
            attention_mask=padding_mask,
        )
        out_2 = self.layernorm_2(out_1 + attention_output_2)

        proj_output = self.dense_proj(out_2)
        return self.layernorm_3(out_2 + proj_output)

    def get_causal_attention_mask(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size, sequence_length = input_shape[0], input_shape[1]
        i = tf.range(sequence_length)[:, tf.newaxis]
        j = tf.range(sequence_length)
        mask = tf.cast(i >= j, dtype="int32")
        mask = tf.reshape(mask, (1, input_shape[1], input_shape[1]))
        mult = tf.concat(
            [tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)],
            axis=0,
        )
        return tf.tile(mask, mult)

embed_dim = 128
latent_dim = 8192
num_heads = 8

encoder_inputs = keras.Input(shape=(None,), dtype="int64", name="encoder_inputs")
x = PositionalEmbedding(sequence_length, vocab_size_notes, embed_dim)(encoder_inputs)
encoder_outputs = TransformerEncoder(embed_dim, latent_dim, num_heads)(x)
encoder = keras.Model(encoder_inputs, encoder_outputs)

decoder_inputs = keras.Input(shape=(None,), dtype="int64", name="decoder_inputs")
encoded_seq_inputs = keras.Input(shape=(None, embed_dim), name="decoder_state_inputs")
x = PositionalEmbedding(sequence_length, 1, embed_dim)(decoder_inputs)
x = TransformerDecoder(embed_dim, latent_dim, num_heads)(x, encoded_seq_inputs)
x = layers.Dropout(0.5)(x)
decoder_outputs = layers.Dense(1, activation="sigmoid")(x)
x2 = layers.Reshape((-1, 1))(decoder_outputs)
decoder = keras.Model([decoder_inputs, encoded_seq_inputs], x2)
decoder_outputs = decoder([decoder_inputs, encoder_outputs])
#decoder_outputs = layers.Dense(20, activation="sigmoid")(decoder_outputs)



max_decoded_sentence_length = 200


def decode_sequence(input_sentence):
    tokenized_input_sentence = notes_vectorization([input_sentence])
    #print(tokenized_input_sentence)
    output_vols = [np.zeros(201)]
    for i in range(max_decoded_sentence_length):
        predictions = transformer([tokenized_input_sentence, tf.constant([output_vols[0][:-1]])])
        #print(float(predictions[0, i, 0]))
        output_vols[0][i+1]=(round(float(predictions[0, i, 0]), 2))


        #sampled_token_index = np.argmax(predictions[0, i, :])
        #sampled_token = vols_index_lookup[sampled_token_index]
        #decoded_sentence += " " + sampled_token

        #if float(predictions[0, i, 0]) == 0:
        #    break
    return output_vols


train = True

if train:
    test_notes_texts = [pair[0] for pair in test_pairs]
    train_notes_texts = [pair[0] for pair in train_pairs]

    transformer = keras.Model(
        [encoder_inputs, decoder_inputs], decoder_outputs, name="transformer"
    )

    epochs = 2

    transformer.summary()
    
    #keras.utils.plot_model(transformer, show_shapes=True, show_dtype=True, expand_nested=True, show_layer_activations=True, show_trainable=True)

    def custom_loss(y_true, y_pred):
        print(y_true)
        print(y_pred)
        
        print(K.mean(K.square(y_pred - y_true), axis=-1))

        x = tf.experimental.numpy.diff(y_pred, axis=-1)
        y = tf.experimental.numpy.diff(y_true, axis=-1)
        z = K.mean(K.square(x - y))

        print(z + K.mean(K.square(y_pred - y_true), axis=-1))
        
        return z + K.mean(K.square(y_pred - y_true), axis=-1)

    
    transformer.compile(
        keras.optimizers.experimental.AdamW(learning_rate=.0002), loss="mean_squared_error", metrics=["cosine_similarity"]
    )

    plotter = PlotLearning()

    transformer.fit(train_ds, epochs=epochs, validation_data=val_ds, callbacks=[plotter], verbose=2)
            
    transformer.save_weights("s2s/weights")




transformer = transformer = keras.Model(
    [encoder_inputs, decoder_inputs], decoder_outputs, name="transformer"
)
transformer.load_weights("s2s/weights")



test_notes_texts = [pair[0] for pair in test_pairs]
for k in range(0):
    notes = ""
    for j in range(random.randint(1,3)):
        x = random.randint(0, len(test_pairs)-1)
        notes += test_pairs[x][0] + " "
    notes = notes[:-1]
    translated = decode_sequence(notes)
    translated = translated[0][:len(notes.split(" "))+1]
    print(f"IN:  {notes}")
    print(f"OUT: {' '.join(str(int(x*100)) for i, x in enumerate(translated) if i > 0)}")

plotter.finish("filename")