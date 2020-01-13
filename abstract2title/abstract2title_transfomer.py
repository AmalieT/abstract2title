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
from keras_utils import BatchCheckpoint, BatchEarlyStopping, DecodeVal, transformer, CustomSchedule, sparse_cross_entropy, predict
import math
import tensorflow as tf
import tensorflow_probability as tfp

batch_size = 256
epochs = 15
title_maxlen = 32
abstract_maxlen = 256

word2index = pickle.load(open(os.path.join("data", 'word2index.pkl'), 'rb'))
index2word = pickle.load(open(os.path.join("data", 'index2word.pkl'), 'rb'))

datapath = os.path.join("data", "abstract2title.tfrecord")
validation_datapath = os.path.join(
    "data", "abstract2title_val.tfrecord")


def feature_label_pairs(example):
  parsed_example = tf.io.parse_single_example(
      example, {
          'abstract': tf.io.FixedLenFeature(shape=(abstract_maxlen), dtype=tf.int64),
          'title': tf.io.FixedLenFeature(shape=(title_maxlen), dtype=tf.int64),
          # I made a typo and now it's baked in lmao
          'titlte_out': tf.io.FixedLenFeature(shape=(title_maxlen), dtype=tf.int64)
      })
  return {"inputs": parsed_example['abstract'], "dec_inputs": parsed_example['title']}, parsed_example['titlte_out']


dataset = tf.data.TFRecordDataset(
    [datapath]).map(feature_label_pairs, num_parallel_calls=tf.data.experimental.AUTOTUNE).shuffle(10 * batch_size).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

validation_dataset = tf.data.TFRecordDataset(
    [validation_datapath]).map(feature_label_pairs, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE).cache()

vocab_size = len(word2index.items())

learning_rate = CustomSchedule()

# optimizer = tf.keras.optimizers.Adam(
# learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

optimizer = tf.keras.optimizers.SGD(learning_rate=0.0001)


def accuracy(y_true, y_pred):
  y_true = tf.reshape(y_true, shape=(-1, title_maxlen))
  return tf.keras.metrics.sparse_categorical_accuracy(y_true, y_pred)


mirrored_strategy = tf.distribute.MirroredStrategy()

with mirrored_strategy.scope():
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
    "data", 'abstract2title_transformer_checkpoint.{epoch:02d}-{val_loss:.2f}.keras')

try:
  model.load_weights(path_checkpoint)
except Exception as error:
  print("Error trying to load checkpoint.")
  print(error)


callback_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    path_checkpoint, save_weights_only=True, monitor='val_loss')


callback_early_stopping = EarlyStopping(monitor='val_loss',
                                        patience=3, verbose=1)

# callback_decode_val = DecodeVal(model=model, eval_every=3000, decode_function=predict,
# validation_data=validation_dataset, word2index=word2index, index2word=index2word, n_eval=1, beam_width=2)


callbacks = [callback_early_stopping,
             callback_checkpoint,
             callback_checkpoint_backup, ]
# callback_decode_val]

model.fit(dataset,
          epochs=epochs,
          shuffle='batch',
          callbacks=callbacks,
          validation_data=validation_dataset
          )
