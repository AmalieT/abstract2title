#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import os
import string
import pickle
import operator
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Lambda, Bidirectional, Concatenate, Activation
import numpy as np
from tensorflow.keras.utils import HDF5Matrix
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from keras_utils import BatchCheckpoint, BatchEarlyStopping, DecodeVal, transformer
import math
import tensorflow as tf


batch_size = 128
epochs = 10
title_maxlen = 32
abstract_maxlen = 256

train_size = 4432316
train_end = math.floor(train_size / batch_size) * batch_size

word2index = pickle.load(open(os.path.join("data", 'word2index.pkl'), 'rb'))
index2word = pickle.load(open(os.path.join("data", 'index2word.pkl'), 'rb'))

datapath = os.path.join("data", "abstract2title.hdf5")
max_vocab_size = len(word2index.items())


model = transformer(
    vocab_size=max_vocab_size,
    num_layers=2,
    units=512,
    d_model=256,
    num_heads=8,
    dropout=0.1)


def sparse_cross_entropy(y_true, y_pred):
  y_true = tf.reshape(y_true, shape=(-1, title_maxlen))

  loss = tf.keras.losses.SparseCategoricalCrossentropy(
      from_logits=True, reduction='none')(y_true, y_pred)

  # mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
  # loss = tf.multiply(loss, mask)

  return tf.reduce_mean(loss)


model.compile(optimizer='rmsprop', loss=sparse_cross_entropy)
model.summary()


def evaluate(sentence, beam_width=5):
  initial_output = [word2index['<BOS>']]
  beams = [(initial_output, 0)]

  sentence = tf.expand_dims(sentence, axis=0)

  isComplete = False

  while not isComplete:
    new_beams = []
    for beam in beams:
      if len(beam[0]) > title_maxlen or beam[0][-1] == word2index['<EOS>']:
        isComplete = True
      output = tf.expand_dims(beam[0], 0)
      predictions = model(
          inputs=[sentence, output], training=False)

      predictions = tf.nn.softmax(predictions[:, -1:, :])

      predictions = tf.squeeze(predictions)

      top_k_indices = tf.argsort(predictions, axis=-1, direction='DESCENDING')

      for ind in top_k_indices.numpy()[:beam_width]:
        prob = math.log(predictions[ind].numpy()) + beam[1]
        new_output = beam[0] + [ind]
        new_beams.append((new_output, prob))

    beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]

  return beams


def predict(sentence, beam_width=5, n_beams=2):
  beams = evaluate(sentence, beam_width=beam_width)

  predictions = []
  for beam in beams:
    predicted_sentence = " ".join([index2word[t] for t in beam[0]])
    predictions.append((predicted_sentence, math.exp(beam[1])))

  predictions = sorted(predictions, key=lambda x: x[1], reverse=True)[:n_beams]

  return predictions


encoder_input_data = HDF5Matrix(datapath, 'abstracts_train_tokens', 0,
                                None)

decoder_input_data = HDF5Matrix(datapath, 'titles_train_tokens', 0,
                                None)

decoder_output_data = HDF5Matrix(
    datapath, 'titles_train_tokens_output', 0, None)


encoder_input_data_test = HDF5Matrix(
    datapath, 'abstracts_test_tokens', 0, batch_size)

decoder_input_data_test = HDF5Matrix(datapath, 'titles_test_tokens', 0,
                                     batch_size)

decoder_output_data_test = HDF5Matrix(
    datapath, 'titles_test_tokens_output', 0, batch_size)

val_size = len(decoder_input_data_test)
path_checkpoint = 'abstract2title_transformer_checkpoint.keras'

path_checkpoint_backup = 'abstract2title_transformer_checkpoint_backup.keras'

try:
  model.load_weights(path_checkpoint)
except Exception as error:
  print("Error trying to load checkpoint.")
  print(error)


callback_checkpoint = BatchCheckpoint(name=path_checkpoint, save_every=1000)

callback_checkpoint_backup = BatchCheckpoint(
    name=path_checkpoint_backup, save_every=1001)

callback_early_stopping = EarlyStopping(monitor='val_loss',
                                        patience=3, verbose=1)

callback_decode_val = DecodeVal(eval_every=1000, decode_function=predict,
                                validation_inputs=encoder_input_data_test, validation_outputs=decoder_input_data_test, index2word=index2word, n_eval=1, beam_width=10)


callbacks = [callback_early_stopping,
             callback_checkpoint,
             callback_checkpoint_backup,
             callback_decode_val]

model.fit([encoder_input_data, decoder_input_data],
          decoder_output_data,
          batch_size=batch_size,
          epochs=epochs,
          shuffle='batch',
          callbacks=callbacks,
          validation_data=(
    [encoder_input_data_test, decoder_input_data_test], decoder_output_data_test),
    validation_steps=val_size
)
