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
import tensorflow_probability as tfp

batch_size = 128
epochs = 15
keyword_maxlen = 64
abstract_maxlen = 256

# train_size = 4432316
# train_end = math.floor(train_size / batch_size) * batch_size

word2index = pickle.load(
    open(os.path.join("data", 'word2index_a2k.pkl'), 'rb'))
index2word = pickle.load(
    open(os.path.join("data", 'index2word_a2k.pkl'), 'rb'))

datapath = os.path.join("data", "abstract2keyword.hdf5")
max_vocab_size = len(word2index.items())


model = transformer(
    vocab_size=max_vocab_size,
    num_layers=2,
    units=512,
    d_model=256,
    num_heads=8,
    dropout=0.1)

# model = transformer(
#     vocab_size=max_vocab_size,
#     num_layers=1,
#     units=64,
#     d_model=32,
#     num_heads=8,
#     dropout=0.1)


def sparse_cross_entropy(y_true, y_pred):
  y_true = tf.reshape(y_true, shape=(-1, keyword_maxlen))

  loss = tf.keras.losses.SparseCategoricalCrossentropy(
      from_logits=True, reduction='none')(y_true, y_pred)

  mask = tf.cast(tf.not_equal(y_true, 2), tf.float32)
  loss = tf.multiply(loss, mask)

  return tf.reduce_mean(loss)


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

  def __init__(self, d_model=256, warmup_steps=4000):
    super(CustomSchedule, self).__init__()

    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)

    self.warmup_steps = warmup_steps

  def __call__(self, step):
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps**-1.5)

    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


learning_rate = CustomSchedule()

optimizer = tf.keras.optimizers.Adam(
    learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

# optimizer = tf.keras.optimizers.SGD(learning_rate=0.0001)


def accuracy(y_true, y_pred):
  y_true = tf.reshape(y_true, shape=(-1, keyword_maxlen))
  return tf.keras.metrics.sparse_categorical_accuracy(y_true, y_pred)


model.compile(optimizer=optimizer,
              loss=sparse_cross_entropy, metrics=[accuracy])
model.summary()


def sample_without_replacement(logits, K):
  """
  Sample from a tensor of logits without replacement using the Gumbel max trick

  https://arxiv.org/pdf/1903.06059.pdf
  http://timvieira.github.io/blog/post/2014/08/01/gumbel-max-trick-and-weighted-reservoir-sampling/
  This trick is so fucking cool. Who comes up with this shit?

  Note that the distribution doesn't depend on logits or K, meaning no costly distribution instantiations
  The world is your oyster with the Gumbel max trick.
  """
  dist = tfp.distributions.Gumbel(loc=0, scale=1)
  z = dist.sample(tf.shape(logits))
  _, indices = tf.nn.top_k(logits + z, K)
  return indices


def stochastic_beam_search(sentence, beam_width=5):
  # Make a prediction using a stochastic beam search
  initial_output = [word2index['<BOS>']]
  beams = [(initial_output, 0)]

  sentence = tf.expand_dims(sentence, axis=0)

  isComplete = False

  while not isComplete:
    new_beams = []
    for beam in beams:
      if len(beam[0]) > keyword_maxlen:
        isComplete = True
      if beam[0][-1] == word2index['<EOS>']:
        new_beams.append(beam)
        continue
      output = tf.expand_dims(beam[0], 0)
      predictions = model(
          inputs=[sentence, output], training=False)

      predictions = tf.nn.log_softmax(predictions[:, -1:, :])

      predictions = tf.squeeze(predictions)

      top_k_indices = sample_without_replacement(predictions, beam_width)

      for ind in top_k_indices.numpy()[:beam_width]:
        prob = predictions[ind].numpy() + beam[1]
        new_output = beam[0] + [ind]
        new_beams.append((new_output, prob))

    beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]
    if all([beam[0][-1] == word2index['<EOS>'] for beam in beams]):
      isComplete = True

  return beams


def predict(sentence, beam_width=5, n_beams=2):
  beams = stochastic_beam_search(sentence, beam_width=beam_width)

  predictions = []
  for beam in beams:
    predicted_sentence = " ".join([index2word[t] for t in beam[0]])
    predictions.append((predicted_sentence, math.exp(beam[1])))

  predictions = sorted(predictions, key=lambda x: x[1], reverse=True)[:n_beams]

  return predictions


encoder_input_data = HDF5Matrix(datapath, 'abstracts_train_tokens', 0,
                                None)

decoder_input_data = HDF5Matrix(datapath, 'keywords_train_tokens', 0,
                                None)

decoder_output_data = HDF5Matrix(
    datapath, 'keywords_train_tokens_output', 0, None)


encoder_input_data_test = HDF5Matrix(
    datapath, 'abstracts_test_tokens', 0, None)

decoder_input_data_test = HDF5Matrix(datapath, 'keywords_test_tokens', 0,
                                     None)

decoder_output_data_test = HDF5Matrix(
    datapath, 'keywords_test_tokens_output', 0, None)

val_size = len(decoder_input_data_test)
path_checkpoint = 'abstract2keyword_transformer_checkpoint.keras'

path_checkpoint_backup = 'abstract2keyword_transformer_checkpoint_backup.keras'

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

callback_decode_val = DecodeVal(eval_every=3000, decode_function=predict,
                                validation_inputs=encoder_input_data_test, validation_outputs=decoder_input_data_test, index2word=index2word, n_eval=1, beam_width=2)


callbacks = [callback_early_stopping,
             callback_checkpoint,
             callback_checkpoint_backup, ]
# callback_decode_val]

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
