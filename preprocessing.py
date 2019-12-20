import h5py
from vocab_utils import create_bpe_vocab, bpe_to_index, tokenize_corpus
import pickle
import sys
import os
from numpy.random import shuffle
import numpy as np

titles_train_raw = os.path.join("data", "titles_train.txt")
titles_train = os.path.join("data", "titles_train_tokens.bpe")
abstracts_train_raw = os.path.join("data", "abstracts_train.txt")
abstracts_train = os.path.join("data", "abstracts_train_tokens.bpe")
hdf5_filename = os.path.join("data", "abstract2title.hdf5")


def write_tokenized_corpus():
    tokenize_corpus(titles_train_raw, titles_train)
    tokenize_corpus(abstracts_train_raw, abstracts_train)


def write_bpe_vocab():
    word2index, index2word = create_bpe_vocab(titles_train, abstracts_train)
    pickle.dump(word2index, open(os.path.join("data", 'word2index.pkl'), 'wb'))
    pickle.dump(index2word, open(os.path.join("data", 'index2word.pkl'), 'wb'))


def write_train_hdf5():
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
    n_titles_test = int(file_len(titles_train) * validation_fraction)
    n_titles_train = file_len(titles_train) - n_titles_test
    shuffled = np.arange(n_titles_train)
    shuffle(shuffled)

    with h5py.File(hdf5_filename, "w") as f:
        titles_train_tokens_dset = f.create_dataset(
            "titles_train_tokens", (n_titles_train, title_maxlen), dtype='i8')

        titles_train_tokens_dset[:, :] = 1

        titles_train_tokens_output_dset = f.create_dataset(
            "titles_train_tokens_output", (n_titles_train, title_maxlen), dtype='i8')

        titles_train_tokens_output_dset[:, :] = 1

        titles_test_tokens_dset = f.create_dataset(
            "titles_test_tokens", (n_titles_test, title_maxlen), dtype='i8')

        titles_test_tokens_dset[:, :] = 1

        titles_test_tokens_output_dset = f.create_dataset(
            "titles_test_tokens_output", (n_titles_test, title_maxlen), dtype='i8')

        titles_test_tokens_output_dset[:, :] = 1

        abstracts_train_tokens_dset = f.create_dataset(
            "abstracts_train_tokens", (n_titles_train, abstract_maxlen), dtype='i8')

        abstracts_train_tokens_dset[:, :] = 1

        abstracts_test_tokens_dset = f.create_dataset(
            "abstracts_test_tokens", (n_titles_test, abstract_maxlen), dtype='i8')

        abstracts_test_tokens_dset[:, :] = 1

        with open(titles_train, 'r') as f:
            for i, l in enumerate(f):
                if i % 10000 == 0:
                    sys.stdout.write("Processing title: %d   \r" % (i))
                    sys.stdout.flush()

                indexed = bpe_to_index(
                    l, word2index, title_maxlen)

                if i < n_titles_train:
                    titles_train_tokens_dset[shuffled[i], :len(
                        indexed)] = indexed
                    titles_train_tokens_output_dset[shuffled[i], :len(
                        indexed) - 1] = indexed[1:]
                else:
                    titles_test_tokens_dset[i -
                                            n_titles_train, :len(indexed)] = indexed
                    titles_test_tokens_output_dset[i - n_titles_train, :len(
                        indexed) - 1] = indexed[1:]

        with open(abstracts_train, 'r') as f:
            for i, l in enumerate(f):
                if i % 10000 == 0:
                    sys.stdout.write("Processing abstract: %d   \r" % (i))
                    sys.stdout.flush()

                tokens = l.split()
                if i < n_titles_train:
                    abstracts_train_tokens_dset[shuffled[i], :len(tokens) + 2] = bpe_to_index(
                        l, word2index, abstract_maxlen)
                else:
                    abstracts_test_tokens_dset[i - n_titles_train, :len(tokens) + 2] = bpe_to_index(
                        l, word2index, abstract_maxlen)


def main():
    if sys.argv[1] == "write_tokenized_corpus":
        write_tokenized_corpus()
    elif sys.argv[1] == "write_bpe_vocab":
        write_bpe_vocab()
    elif sys.argv[1] == "write_train_hdf5":
        write_train_hdf5()


if __name__ == "__main__":
    main()
