import h5py
from vocab_utils import create_bpe_vocab, bpe_to_index, tokenize_corpus
import pickle
import sys
import os
from numpy.random import shuffle
import numpy as np

keywords_train_raw = os.path.join("data", "keywords_train_a2k.txt")
keywords_train_tokens = os.path.join("data", "keywords_train_a2k_tokens.txt")
keywords_train = os.path.join("data", "keywords_train_a2k_tokens.bpe")

keywords_test_raw = os.path.join("data", "keywords_test_a2k.txt")
keywords_test = os.path.join("data", "keywords_test_a2k_tokens.txt")

abstracts_train_raw = os.path.join("data", "abstracts_train_a2k.txt")
abstracts_train_tokens = os.path.join("data", "abstracts_train_a2k_tokens.txt")
abstracts_train = os.path.join("data", "abstracts_train_a2k_tokens.bpe")

abstracts_test_raw = os.path.join("data", "abstracts_test_a2k.txt")
abstracts_test = os.path.join("data", "abstracts_test_a2k_tokens.txt")

hdf5_filename = os.path.join("data", "abstract2keyword.hdf5")


def write_tokenized_corpus():
    tokenize_corpus(keywords_train_raw, keywords_train_tokens)
    tokenize_corpus(keywords_test_raw, keywords_test)

    tokenize_corpus(abstracts_train_raw, abstracts_train_tokens)
    tokenize_corpus(abstracts_test_raw, abstracts_test)


def write_bpe_vocab():
    word2index, index2word = create_bpe_vocab(abstracts_train, keywords_train)
    pickle.dump(word2index, open(os.path.join(
        "data", 'word2index_a2k.pkl'), 'wb'))
    pickle.dump(index2word, open(os.path.join(
        "data", 'index2word_a2k.pkl'), 'wb'))


def write_train_hdf5():
    keyword_maxlen = 64
    abstract_maxlen = 256
    validation_fraction = 0.0005

    word2index = pickle.load(
        open(os.path.join("data", 'word2index_a2k.pkl'), 'rb'))
    index2word = pickle.load(
        open(os.path.join("data", 'index2word_a2k.pkl'), 'rb'))

    vocab_size = len(word2index.items())

    def file_len(fname):
        with open(fname) as f:
            for i, l in enumerate(f):
                pass
        return i + 1

    n_keywords = file_len(keywords_train)
    n_keywords_test = int(file_len(keywords_train) * validation_fraction)
    n_keywords_train = file_len(keywords_train) - n_keywords_test
    shuffled = np.arange(n_keywords_train)
    shuffle(shuffled)

    with h5py.File(hdf5_filename, "w") as f:
        keywords_train_tokens_dset = f.create_dataset(
            "keywords_train_tokens", (n_keywords_train, keyword_maxlen), dtype='i8')

        keywords_train_tokens_dset[:, :] = 2

        keywords_train_tokens_output_dset = f.create_dataset(
            "keywords_train_tokens_output", (n_keywords_train, keyword_maxlen), dtype='i8')

        keywords_train_tokens_output_dset[:, :] = 2

        keywords_test_tokens_dset = f.create_dataset(
            "keywords_test_tokens", (n_keywords_test, keyword_maxlen), dtype='i8')

        keywords_test_tokens_dset[:, :] = 2

        keywords_test_tokens_output_dset = f.create_dataset(
            "keywords_test_tokens_output", (n_keywords_test, keyword_maxlen), dtype='i8')

        keywords_test_tokens_output_dset[:, :] = 2

        abstracts_train_tokens_dset = f.create_dataset(
            "abstracts_train_tokens", (n_keywords_train, abstract_maxlen), dtype='i8')

        abstracts_train_tokens_dset[:, :] = 2

        abstracts_test_tokens_dset = f.create_dataset(
            "abstracts_test_tokens", (n_keywords_test, abstract_maxlen), dtype='i8')

        abstracts_test_tokens_dset[:, :] = 2

        with open(keywords_train, 'r') as f:
            for i, l in enumerate(f):
                if i % 10000 == 0:
                    sys.stdout.write("Processing keyword: %d   \r" % (i))
                    sys.stdout.flush()

                indexed = bpe_to_index(
                    l, word2index, keyword_maxlen)

                if i < n_keywords_train:
                    keywords_train_tokens_dset[shuffled[i], :len(
                        indexed)] = indexed
                    keywords_train_tokens_output_dset[shuffled[i], :len(
                        indexed) - 1] = indexed[1:]
                else:
                    keywords_test_tokens_dset[i -
                                              n_keywords_train, :len(indexed)] = indexed
                    keywords_test_tokens_output_dset[i - n_keywords_train, :len(
                        indexed) - 1] = indexed[1:]

        with open(abstracts_train, 'r') as f:
            for i, l in enumerate(f):
                if i % 10000 == 0:
                    sys.stdout.write("Processing abstract: %d   \r" % (i))
                    sys.stdout.flush()

                tokens = l.split()
                if i < n_keywords_train:
                    abstracts_train_tokens_dset[shuffled[i], :len(tokens) + 2] = bpe_to_index(
                        l, word2index, abstract_maxlen)
                else:
                    abstracts_test_tokens_dset[i - n_keywords_train, :len(tokens) + 2] = bpe_to_index(
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
