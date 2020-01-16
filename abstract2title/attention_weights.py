#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import json
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
from keras_utils import BatchCheckpoint, BatchEarlyStopping, DecodeVal, transformer, CustomSchedule, sparse_cross_entropy, predict
import math
import tensorflow as tf
import tensorflow_probability as tfp
import pathlib
import tensorflow.keras.backend as K

base_path = os.path.join(pathlib.Path().absolute(), "data")
batch_size = 256
epochs = 15
title_maxlen = 32
abstract_maxlen = 256

train_size = 4432384
val_size = 375

word2index = pickle.load(open(os.path.join(base_path, 'word2index.pkl'), 'rb'))
index2word = pickle.load(open(os.path.join(base_path, 'index2word.pkl'), 'rb'))

validation_datapath = os.path.join(
    base_path, "abstract2title_val.tfrecord")


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


def stochastic_beam_search(model, sentence, maxlen, beam_width=5, bos_ind=0, eos_ind=1):
  # Make a prediction using a stochastic beam search
  initial_output = [bos_ind]
  beams = [(initial_output, 0, [])]

  sentence = tf.expand_dims(sentence, axis=0)

  isComplete = False

  while not isComplete:
    new_beams = []
    for beam in beams:
      if len(beam[0]) > maxlen:
        isComplete = True
      if beam[0][-1] == eos_ind:
        new_beams.append(beam)
        continue
      output = tf.expand_dims(beam[0], 0)

      def get_predictions(sentence, output):
        predictions, attention_weights = model(
            inputs=[sentence, output])
        return predictions, attention_weights

      predictions, attention_weights = get_predictions(sentence, output)
      attention_weights = tf.squeeze(tf.squeeze(
          tf.reduce_sum(attention_weights, 1), axis=0)[-1:, :])
      attention_weights = tf.nn.softmax(attention_weights, axis=-1)
      attention_weights = list(attention_weights.numpy())
      predictions = tf.nn.log_softmax(predictions[:, -1:, :])

      predictions = tf.squeeze(predictions)

      top_k_indices = sample_without_replacement(predictions, beam_width)

      for ind in top_k_indices.numpy()[:beam_width]:
        prob = predictions[ind].numpy() + beam[1]
        new_output = beam[0] + [ind]
        new_beams.append((new_output, prob, beam[2] + [attention_weights]))

    beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[
        :beam_width]
    if all([beam[0][-1] == eos_ind for beam in beams]):
      isComplete = True

  return beams


def predict_with_attention_weights(model, sentence, word2index, index2word, beam_width=5, n_beams=1, maxlen=32):
  beams = stochastic_beam_search(model,
                                 sentence, maxlen=maxlen, beam_width=beam_width, bos_ind=word2index['<BOS>'], eos_ind=word2index['<EOS>'])

  predictions = []
  for beam in beams:
    predicted_sentence = " ".join([index2word[t] for t in beam[0]])
    detokenized_sentence = " ".join([index2word[t] for t in sentence.numpy()])
    predictions.append(
        (predicted_sentence, detokenized_sentence, math.exp(beam[1]), [[float(y) for y in x] for x in beam[2]]))

  predictions = sorted(predictions, key=lambda x: x[2], reverse=True)[
      :n_beams]

  return predictions


def feature_label_pairs(example):
  parsed_example = tf.io.parse_single_example(
      example, {
          'abstract': tf.io.FixedLenFeature(shape=(abstract_maxlen), dtype=tf.int64),
          'title': tf.io.FixedLenFeature(shape=(title_maxlen), dtype=tf.int64),
          # I made a typo and now it's baked in lmao
          'titlte_out': tf.io.FixedLenFeature(shape=(title_maxlen), dtype=tf.int64)
      })
  return {"inputs": parsed_example['abstract'], "dec_inputs": parsed_example['title']}, parsed_example['titlte_out']


validation_dataset = tf.data.TFRecordDataset(
    [validation_datapath]).map(feature_label_pairs, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(1).prefetch(tf.data.experimental.AUTOTUNE).repeat()

vocab_size = len(word2index.items())

learning_rate = CustomSchedule()

# optimizer = tf.keras.optimizers.Adam(
# learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

optimizer = tf.keras.optimizers.SGD(learning_rate=0.0001)


def accuracy(y_true, y_pred):
  y_true = tf.reshape(y_true, shape=(-1, title_maxlen))
  return tf.keras.metrics.sparse_categorical_accuracy(y_true, y_pred)


model = transformer(
    vocab_size=vocab_size,
    num_layers=2,
    units=512,
    d_model=256,
    num_heads=8,
    dropout=0.1)
model.compile(optimizer=optimizer,
              loss=sparse_cross_entropy, metrics=[accuracy])
model.summary()
path_checkpoint = os.path.join(
    base_path, 'abstract2title_transformer_checkpoint_2layer.keras')

model.load_weights(path_checkpoint)

encoder_attention_weights = model.get_layer(
    'decoder').get_layer('decoder_layer_0').get_layer('attention_2').attention_weights

attention_weights = K.function(
    model.inputs, [model.output, encoder_attention_weights])

decoded_with_attention = []
i = 0
with open("attention_weights_visual.json", 'w') as f:

  for example in validation_dataset.take(2000):
    print(i)
    i += 1
    prediction_with_attention = predict_with_attention_weights(
        attention_weights, tf.squeeze(example[0]['inputs']), word2index, index2word, beam_width=5, n_beams=1, maxlen=32)

    f.write(json.dumps(prediction_with_attention) + "\n")
