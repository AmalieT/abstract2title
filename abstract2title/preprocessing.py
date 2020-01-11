import h5py
from vocab_utils import create_bpe_vocab, bpe_to_index, tokenize_corpus
import pickle
import sys
import os
from numpy.random import shuffle
import numpy as np
import tensorflow as tf
import itertools

titles_train_raw = os.path.join("data", "titles_train.txt")
titles_train_tokens = os.path.join("data", "titles_train_tokens.txt")

titles_train = os.path.join("data", "titles_train_tokens.bpe")
abstracts_train_raw = os.path.join("data", "abstracts_train.txt")
abstracts_train_tokens = os.path.join("data", "abstracts_train_tokens.txt")

abstracts_train = os.path.join("data", "abstracts_train_tokens.bpe")
titles_test_raw = os.path.join("data", "titles_test.txt")
titles_test = os.path.join("data", "titles_test_tokens.txt")
abstracts_test_raw = os.path.join("data", "abstracts_test.txt")
abstracts_test = os.path.join("data", "abstracts_test_tokens.txt")

tfrecord_filename = os.path.join("data", "abstract2title.tfrecord")
tfrecord_validation_filename = os.path.join(
    "data", "abstract2title_val.tfrecord")


def write_tokenized_corpus():
    tokenize_corpus(titles_train_raw, titles_train_tokens)
    tokenize_corpus(abstracts_train_raw, abstracts_train_tokens)
    tokenize_corpus(titles_test_raw, titles_test)
    tokenize_corpus(abstracts_test_raw, abstracts_test)


def write_bpe_vocab():
    word2index, index2word = create_bpe_vocab(titles_train, abstracts_train)
    pickle.dump(word2index, open(os.path.join("data", 'word2index.pkl'), 'wb'))
    pickle.dump(index2word, open(os.path.join("data", 'index2word.pkl'), 'wb'))


def write_train_tfrecord():
    title_maxlen = 32
    abstract_maxlen = 256
    validation_fraction = 0.0001

    word2index = pickle.load(
        open(os.path.join("data", 'word2index.pkl'), 'rb'))
    index2word = pickle.load(
        open(os.path.join("data", 'index2word.pkl'), 'rb'))

    vocab_size = len(word2index.items())

    def file_len(fname):
        with open(fname) as f:
            for i, l in enumerate(f):
                pass
        return i + 1

    n_titles = file_len(titles_train)
    n_titles_validation = int(file_len(titles_train) * validation_fraction)
    n_titles_train = file_len(titles_train) - n_titles_validation

    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    def serialize_train_example(abstract, title, title_out):
        """
        Creates a tf.Example message ready to be written to a file.
        """
        # Create a dictionary mapping the feature name to the tf.Example-compatible
        # data type.
        feature = {
            'abstract': _int64_feature(abstract),
            'title': _int64_feature(title),
            'titlte_out': _int64_feature(title_out)
        }

        # Create a Features message using tf.train.Example.

        example_proto = tf.train.Example(
            features=tf.train.Features(feature=feature))
        return example_proto.SerializeToString()

    def clean_index_pad(title, abstract):
        abstract_clean = abstract.strip()
        abstract_indexed = bpe_to_index(
            abstract_clean, word2index, abstract_maxlen)

        abstract_padded = np.zeros(
            (abstract_maxlen), dtype=int)
        # 2 = <PAD>
        abstract_padded[:] = 2
        abstract_padded[:len(abstract_indexed)
                        ] = abstract_indexed

        title_clean = title.strip()
        title_indexed = bpe_to_index(
            title_clean, word2index, title_maxlen)

        title_padded = np.zeros((title_maxlen), dtype=int)
        title_out_padded = np.zeros((title_maxlen), dtype=int)
        # 2 = <PAD>
        title_padded[:] = 2
        title_out_padded[:] = 2

        title_padded[:len(title_indexed)] = title_indexed

        # Out: shift target by one
        title_out_padded[:len(title_indexed) -
                         1] = title_indexed[1:]

        return abstract_padded, title_padded, title_out_padded

    def tf_example_generator_factory(start, stop):
        def tf_example_generator():
            # Generator for serialized example messages from our dataset
            with open(titles_train, 'r') as titles_file:
                with open(abstracts_train, 'r') as abstracts_file:
                    for title, abstract in itertools.islice(zip(titles_file, abstracts_file), start, stop):
                        yield serialize_train_example(*clean_index_pad(title, abstract))
        return tf_example_generator

    serialized_features_dataset = tf.data.Dataset.from_generator(
        tf_example_generator_factory(
            start=0, stop=n_titles_train), output_types=tf.string, output_shapes=())

    writer = tf.data.experimental.TFRecordWriter(tfrecord_filename)
    writer.write(serialized_features_dataset)

    serialized_features_dataset = tf.data.Dataset.from_generator(
        tf_example_generator_factory(
            start=n_titles_train, stop=n_titles), output_types=tf.string, output_shapes=())

    writer = tf.data.experimental.TFRecordWriter(tfrecord_validation_filename)
    writer.write(serialized_features_dataset)


def main():
    if sys.argv[1] == "write_tokenized_corpus":
        write_tokenized_corpus()
    elif sys.argv[1] == "write_bpe_vocab":
        write_bpe_vocab()
    elif sys.argv[1] == "write_train_tfrecord":
        write_train_tfrecord()


if __name__ == "__main__":
    main()
