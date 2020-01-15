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
from keras_utils import BatchCheckpoint, BatchEarlyStopping, DecodeVal, transformer, CustomSchedule, sparse_cross_entropy, predict, encoder_classifier
import math
import tensorflow as tf
import tensorflow_probability as tfp
import pathlib

batch_size = 256
epochs = 15
title_maxlen = 32
abstract_maxlen = 256
base_path = os.path.join(pathlib.Path().absolute(), "data")
word2index = pickle.load(
    open(os.path.join(base_path, 'word2index_a2j.pkl'), 'rb'))
index2word = pickle.load(
    open(os.path.join(base_path, 'index2word_a2j.pkl'), 'rb'))

datapath = os.path.join(base_path, "abstract2journal.tfrecord")
validation_datapath = os.path.join(
    base_path, "abstract2journal_val.tfrecord")


def feature_label_pairs(example):
  parsed_example = tf.io.parse_single_example(
      example, {
          'abstract': tf.io.FixedLenFeature(shape=(abstract_maxlen), dtype=tf.int64),
          'journal': tf.io.FixedLenFeature(shape=(title_maxlen), dtype=tf.int64),
      })
  return parsed_example['abstract'], parsed_example['journal']


dataset = tf.data.TFRecordDataset(
    [datapath]).map(feature_label_pairs, num_parallel_calls=tf.data.experimental.AUTOTUNE).shuffle(10 * batch_size).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

validation_dataset = tf.data.TFRecordDataset(
    [validation_datapath]).map(feature_label_pairs, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(5).prefetch(tf.data.experimental.AUTOTUNE)

vocab_size = len(word2index.items())

learning_rate = CustomSchedule()

optimizer = tf.keras.optimizers.Adam(
    learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

# optimizer = tf.keras.optimizers.SGD(learning_rate=0.0001)


# def accuracy(y_true, y_pred):
# y_true = tf.reshape(y_true, shape=(-1, title_maxlen))
# return tf.keras.metrics.sparse_categorical_accuracy(y_true, y_pred)


mirrored_strategy = tf.distribute.MirroredStrategy()

with mirrored_strategy.scope():
  model = encoder_classifier(
      vocab_size=vocab_size,
      num_layers=2,
      units=512,
      d_model=256,
      num_heads=8,
      dropout=0.1,
      num_classes=1,
      abstract_maxlen=abstract_maxlen)
  model.compile(optimizer=optimizer,
                loss='binary_crossentropy', metrics=['binary_accuracy'])
model.summary()

path_checkpoint = os.path.join(
    base_path, 'abstract2journals_transformer_checkpoint.keras')

# path_checkpoint_backup = 'abstract2journals_transformer_checkpoint_backup.keras'

try:
  model.load_weights(path_checkpoint)
except Exception as error:
  print("Error trying to load checkpoint.")
  print(error)


callback_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    path_checkpoint, save_weights_only=True, monitor='val_loss')
# callback_checkpoint = BatchCheckpoint(name=path_checkpoint, save_every=1000)

# callback_checkpoint_backup = BatchCheckpoint(
#     name=path_checkpoint_backup, save_every=1001)

# callback_early_stopping = EarlyStopping(monitor='val_loss',
# patience=3, verbose=1)

# callback_decode_val = DecodeVal(model=model, eval_every=3000, decode_function=predict,
# validation_data=validation_dataset, word2index=word2index, index2word=index2word, n_eval=1, beam_width=2)


callbacks = [callback_checkpoint,
             callback_checkpoint_backup, ]
# callback_decode_val]

model.fit(dataset,
          epochs=epochs,
          shuffle='batch',
          callbacks=callbacks,
          validation_data=validation_dataset,
          # steps_per_epoch=4888,
          # validation_steps=2
          )
