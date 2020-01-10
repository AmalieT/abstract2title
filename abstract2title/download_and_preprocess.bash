# /bin/bash

# python3.6 download_corpus.py
# python3.6 preprocessing.py write_tokenized_corpus

# subword-nmt learn-joint-bpe-and-vocab --input data/abstracts_train_tokens.txt data/titles_train_tokens.txt -s 20000 -o data/codes.bpe --write-vocabulary data/abstracts_train_vocab.bpe data/titles_train_vocab.bpe

# subword-nmt apply-bpe -c data/codes.bpe --vocabulary data/titles_train_vocab.bpe --vocabulary-threshold 1 < data/titles_train_tokens.txt > data/titles_train_tokens.bpe
# subword-nmt apply-bpe -c data/codes.bpe --vocabulary data/abstracts_train_vocab.bpe --vocabulary-threshold 1 < data/abstracts_train_tokens.txt > data/abstracts_train_tokens.bpe

subword-nmt apply-bpe -c data/codes.bpe --vocabulary data/titles_train_vocab.bpe --vocabulary-threshold 1 < data/titles_test_tokens.txt > data/titles_test_tokens.bpe
subword-nmt apply-bpe -c data/codes.bpe --vocabulary data/abstracts_train_vocab.bpe --vocabulary-threshold 1 < data/abstracts_test_tokens.txt > data/abstracts_test_tokens.bpe

# python3.6 preprocessing.py write_bpe_vocab
# python3.6 preprocessing.py write_train_hdf5
