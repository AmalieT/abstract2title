import h5py
from vocab_utils import create_bpe_vocab, bpe_to_index, tokenize_corpus, journals_to_class
import pickle
import sys
import os
from numpy.random import shuffle
import numpy as np

journals_train = os.path.join("data", "journals_train_a2j.txt")

journals_test = os.path.join("data", "journals_test_a2j.txt")

abstracts_train_raw = os.path.join("data", "abstracts_train_a2j.txt")
abstracts_train_tokens = os.path.join("data", "abstracts_train_a2j_tokens.txt")
abstracts_train = os.path.join("data", "abstracts_train_a2j_tokens.bpe")

abstracts_test_raw = os.path.join("data", "abstracts_test_a2j.txt")
abstracts_test = os.path.join("data", "abstracts_test_a2j_tokens.txt")

hdf5_filename = os.path.join("data", "abstract2journal.hdf5")


def write_tokenized_corpus():
    tokenize_corpus(abstracts_train_raw, abstracts_train_tokens)
    tokenize_corpus(abstracts_test_raw, abstracts_test)


def write_train_hdf5():
    abstract_maxlen = 256
    validation_fraction = 0.0005

    word2index = pickle.load(
        open(os.path.join("data", 'word2index.pkl'), 'rb'))
    index2word = pickle.load(
        open(os.path.join("data", 'index2word.pkl'), 'rb'))

    journal2class = pickle.load(
        open(os.path.join("data", 'journal2class.pkl'), 'rb'))
    n_classes = len(journal2class.items())
    print("N classes: {}".format(n_classes))

    vocab_size = len(word2index.items())

    def file_len(fname):
        with open(fname) as f:
            for i, l in enumerate(f):
                pass
        return i + 1

    n_journals = file_len(journals_train)
    n_journals_test = int(file_len(journals_train) * validation_fraction)
    n_journals_train = file_len(journals_train) - n_journals_test
    shuffled = np.arange(n_journals_train)
    shuffle(shuffled)

    with h5py.File(hdf5_filename, "w") as f:
        journals_train_dset = f.create_dataset(
            "journals_train", (n_journals_train, n_classes), dtype='i8')

        journals_train_dset[:, :] = 0

        journals_test_dset = f.create_dataset(
            "journals_test", (n_journals_test, n_classes), dtype='i8')

        journals_test_dset[:, :] = 0

        abstracts_train_tokens_dset = f.create_dataset(
            "abstracts_train_tokens", (n_journals_train, abstract_maxlen), dtype='i8')

        abstracts_train_tokens_dset[:, :] = 2

        abstracts_test_tokens_dset = f.create_dataset(
            "abstracts_test_tokens", (n_journals_test, abstract_maxlen), dtype='i8')

        abstracts_test_tokens_dset[:, :] = 2

        with open(journals_train, 'r') as f:
            for i, l in enumerate(f):
                if i % 10000 == 0:
                    sys.stdout.write("Processing journal: %d   \r" % (i))
                    sys.stdout.flush()

                journal_class = int(l.strip())
                if i < n_journals_train:
                    journals_train_tokens_dset[shuffled[i], journal_class] = 1
                else:
                    journals_test_tokens_dset[i -
                                              n_journals_train, journal_class] = 1

        with open(abstracts_train, 'r') as f:
            for i, l in enumerate(f):
                if i % 10000 == 0:
                    sys.stdout.write("Processing abstract: %d   \r" % (i))
                    sys.stdout.flush()

                tokens = l.split()
                if i < n_journals_train:
                    abstracts_train_tokens_dset[shuffled[i], :len(tokens) + 2] = bpe_to_index(
                        l, word2index, abstract_maxlen)
                else:
                    abstracts_test_tokens_dset[i - n_journals_train, :len(tokens) + 2] = bpe_to_index(
                        l, word2index, abstract_maxlen)


def main():
    if sys.argv[1] == "write_tokenized_corpus":
        write_tokenized_corpus()
    elif sys.argv[1] == "write_train_hdf5":
        write_train_hdf5()


if __name__ == "__main__":
    main()
