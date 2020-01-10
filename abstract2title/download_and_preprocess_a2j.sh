# /bin/bash

# python3.6 download_journal_abstracts.py
python3.6 preprocessing_a2j.py write_tokenized_corpus

subword-nmt apply-bpe -c data/codes.bpe --vocabulary data/abstracts_train_vocab.bpe --vocabulary-threshold 1 < data/abstracts_train_a2j_tokens.txt > data/abstracts_train_a2j_tokens.bpe

subword-nmt apply-bpe -c data/codes.bpe --vocabulary data/abstracts_train_vocab.bpe --vocabulary-threshold 1 < data/abstracts_test_a2j_tokens.txt > data/abstracts_test_a2j_tokens.bpe

python3.6 preprocessing_a2j.py write_train_hdf5
