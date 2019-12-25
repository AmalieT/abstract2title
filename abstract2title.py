#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import os
import string
import pickle
import operator
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Lambda, Bidirectional, Concatenate, Activation
import numpy as np
import keras
from keras.utils.io_utils import HDF5Matrix
import tensorflow as tf
from keras.callbacks import EarlyStopping, TensorBoard
from keras_utils import sparse_cross_entropy, BatchCheckpoint, BatchEarlyStopping, DecodeVal
import math


batch_size = 32
epochs = 10
latent_dim = 256
title_maxlen = 32
abstract_maxlen = 256

word2index = pickle.load(open(os.path.join("data", 'word2index.pkl'), 'rb'))
index2word = pickle.load(open(os.path.join("data", 'index2word.pkl'), 'rb'))

datapath = os.path.join("data", "abstract2title.hdf5")
max_vocab_size = len(word2index.items())

"""
Training Model
"""

encoder_inputs = Input(shape=(None,), dtype='int32')
encoder_one_hot = Lambda(keras.backend.one_hot,
                         arguments={'num_classes': max_vocab_size},
                         output_shape=(None, max_vocab_size))(encoder_inputs)

encoder = Bidirectional(LSTM(units=latent_dim, return_state=True))
encoder_outputs, forward_h, forward_c, backward_h, backward_c = encoder(
    encoder_one_hot)

state_h = Concatenate()([forward_h, backward_h])
state_c = Concatenate()([forward_c, backward_c])
encoder_states = [state_h, state_c]

decoder_inputs = Input(shape=(None, ), dtype='int32')
decoder_inputs_one_hot = Lambda(keras.backend.one_hot,
                                arguments={'num_classes': max_vocab_size},
                                output_shape=(None, max_vocab_size))(decoder_inputs)

decoder_LSTM = LSTM(units=2 * latent_dim, return_sequences=True,
                    return_state=True)
decoder_outputs, _h, _c = decoder_LSTM(
    decoder_inputs_one_hot, initial_state=encoder_states)
decoder_dense = Dense(units=max_vocab_size, activation="linear")
decoder_outputs = decoder_dense(decoder_outputs)

# Note that we need a tf placeholder target tensor bebcause of the sparse cross-entropy loss
decoder_target = tf.placeholder(dtype='int32', shape=(None, None))

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

model.compile(optimizer='rmsprop', loss=sparse_cross_entropy,
              target_tensors=[decoder_target])

model.summary()


"""
Inference Model
"""
encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_input_h = Input(shape=(2 * latent_dim,))
decoder_state_input_c = Input(shape=(2 * latent_dim,))

decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

decoder_outputs, state_h, state_c = decoder_LSTM(
    decoder_inputs_one_hot, initial_state=decoder_states_inputs)

decoder_states = [state_h, state_c]

decoder_outputs = Activation(activation='softmax')(
    decoder_dense(decoder_outputs))

decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)


def decode_sequence(input_seq):
  # Encode the input as state vectors.
  states_value = encoder_model.predict(input_seq)

  # Generate empty target sequence of length 1.
  target_seq = np.zeros((1, 1))
  # Populate the first character of target sequence with the start character.
  target_seq[0, 0] = word2index['<BOS>']

  # Sampling loop for a batch of sequences
  # (to simplify, here we assume a batch of size 1).
  stop_condition = False
  decoded_sentence = ''
  while not stop_condition:
    output_tokens, h, c = decoder_model.predict(
        [target_seq] + states_value)

    # Sample a token
    sampled_token_index = np.argmax(output_tokens[0, -1, :])
    sampled_token = index2word[sampled_token_index]
    decoded_sentence += " {}".format(sampled_token)

    # Exit condition: either hit max length
    # or find stop character.
    if (sampled_token == '<EOS>' or
            len(decoded_sentence.split()) > title_maxlen):
      stop_condition = True

    # Update the target sequence (of length 1).
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = sampled_token_index

    # Update states
    states_value = [h, c]

  return decoded_sentence


def beam_decode_sequence(input_seq, top_k=5):
  # Encode the input as state vectors.
  states_value = encoder_model.predict(input_seq)

  stop_condition = False
  beams = [([word2index['<BOS>']], 0)]
  while not stop_condition:
    new_beams = []
    for beam in beams:
      target_seq = np.zeros((1, 1))
      target_seq[0, 0] = beam[0][-1]
      try:
        states_value = beam[2]
      except IndexError:
        pass
      output_tokens, h, c = decoder_model.predict(
          [target_seq] + states_value)

      states_value = [h, c]
      top_k_indices = (-output_tokens[0, 0, :]).argsort()[:top_k]
      for ind in top_k_indices:
        if ind == word2index['<EOS>'] or len(beam[0]) > title_maxlen:
          stop_condition = True
        prob = math.log(output_tokens[0, 0, ind]) + beam[1]
        new_target_seq = beam[0] + [ind]
        new_beams.append((new_target_seq, prob, states_value))

    beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:top_k]

  decoded_sequences = []
  for beam in beams:
    decoded_sequence = " ".join([index2word[t] for t in beam[0]])
    decoded_sequences += [(decoded_sequence, math.exp(beam[1]))]

  return decoded_sequences


def restrict_length(maxlen, eos_token_index):
  def restrict_length_normalizer_function(sequence):
    restricted_sequence = []
    for i, seq in enumerate(sequence):
      list_here = []
      for j, s in enumerate(seq):
        if s == eos_token_index or i >= maxlen:
          list_here = np.array(sequence[i][:j + 1])
          list_here[-1] = eos_token_index
          restricted_sequence.append(list_here)
          break
    return np.concatenate(restricted_sequence)

  return restrict_length_normalizer_function


encoder_input_data = HDF5Matrix(datapath, 'abstracts_train_tokens', 0,
                                None)

decoder_input_data = HDF5Matrix(datapath, 'titles_train_tokens', 0,
                                None)

decoder_output_data = HDF5Matrix(
    datapath, 'titles_train_tokens_output', 0, None)


encoder_input_data_test = HDF5Matrix(
    datapath, 'abstracts_test_tokens', 0, None)

decoder_input_data_test = HDF5Matrix(datapath, 'titles_test_tokens', 0,
                                     None)

decoder_output_data_test = HDF5Matrix(
    datapath, 'titles_test_tokens_output', 0, None)

path_checkpoint = 'abstract2title_checkpoint.keras'

try:
  model.load_weights(path_checkpoint)
except Exception as error:
  print("Error trying to load checkpoint.")
  print(error)


callback_checkpoint = BatchCheckpoint(name=path_checkpoint, save_every=1000)

callback_early_stopping = EarlyStopping(monitor='val_loss',
                                        patience=3, verbose=1)

callback_decode_val = DecodeVal(eval_every=100, decode_function=beam_decode_sequence,
                                validation_inputs=encoder_input_data_test, validation_outputs=decoder_input_data_test, index2word=index2word, n_eval=1, beam_width=5)


callback_tensorboard = TensorBoard(log_dir='./abstract2title_logs/',
                                   histogram_freq=0,
                                   write_graph=False)

callbacks = [callback_early_stopping,
             callback_checkpoint,
             callback_tensorboard,
             callback_decode_val]

model.fit([encoder_input_data, decoder_input_data],
          decoder_output_data,
          batch_size=batch_size,
          epochs=epochs,
          shuffle='batch',
          callbacks=callbacks,
          validation_data=(
    [encoder_input_data_test, decoder_input_data_test], decoder_output_data_test),
)
