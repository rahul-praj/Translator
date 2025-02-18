import pandas as pd
import numpy as np
import re
from tensorflow import keras
from keras.layers import Input, LSTM, Dense
from keras.models import Model
from tqdm import tqdm
from text_processing import input_docs, target_docs, input_tokens, target_tokens, num_decoder_tokens, num_encoder_tokens

input_features_dict = {token: i for i, token in enumerate(input_tokens)}
target_features_dict = {token: i for i, token in enumerate(target_tokens)}

# Create dictionary in reverse key-val pair (i, token) for decoding

input_decode_dict = {i: token for token, i in input_features_dict.items()}
target_decode_dict = {i: token for token, i in target_features_dict.items()}

encoder_input = np.zeros(len(input_docs), max_encoder_seq_length, num_encoder_tokens)
decoder_input = np.zeros(len(input_docs), max_decoder_seq_length, num_decoder_tokens)

decoder_target = np.zeros(len(target_docs), max_decoder_seq_length, num_decoder_tokens)

for line, (input_doc, target_doc) in enumerate(zip(input_docs, target_docs)):
     
    for timestep, token in enumerate(input_doc.split()):

        encoder_input[line, timestep, input_features_dict[token]] = 1.

    for timestep, token in enumerate(target_doc.split()):

        decoder_input[line, timestep, target_features_dict[token]] = 1.

# Define an encoder 

encoder_inputs = Input(shape=(None, num_encoder_tokens))

latent_dim = 256
encoder = LSTM(latent_dim, return_state=True)

encoder_model, encoder_hidden, encoder_cell = encoder(encoder_inputs)

encoder_states = [encoder_hidden, encoder_cell]

# Define a decoder

decoder_inputs = Input(shape(None, num_decoder_tokens))
decoder = LSTM(latent_dim, return_sequences=True, return_state=True)

decoder_outputs, _, _ = decoder(decoder_inputs, initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')

decoder_outputs = decoder_dense(decoder_outputs)

# Build the training model

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

print("Model summary: \n")
model.summary()
print("\n")

model.compile(optimizer="rmsprop",
              loss="categorical_crossentropy",
              metrics=["accuracy"])

# Choose the batch size and number of epochs
batch_size=50
epochs=50

model.fit([encoder_input_data, decoder_input_data],
                    decoder_target_data,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_split=0.2)




