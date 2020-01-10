# /bin/bash

# python3.6 download_keyword_abstracts.py
python3.6 preprocessing_a2k.py write_tokenized_corpus

subword-nmt learn-joint-bpe-and-vocab --input data/abstracts_train_a2k_tokens.txt data/keywords_train_a2k_tokens.txt -s 20000 -o data/codes_a2k.bpe --write-vocabulary data/abstracts_train_a2k_vocab.bpe data/keywords_train_a2k_vocab.bpe

subword-nmt apply-bpe -c data/codes_a2k.bpe --vocabulary data/keywords_train_a2k_vocab.bpe --vocabulary-threshold 1 < data/keywords_train_a2k_tokens.txt > data/keywords_train_a2k_tokens.bpe
subword-nmt apply-bpe -c data/codes_a2k.bpe --vocabulary data/abstracts_train_a2k_vocab.bpe --vocabulary-threshold 1 < data/abstracts_train_a2k_tokens.txt > data/abstracts_train_a2k_tokens.bpe

subword-nmt apply-bpe -c data/codes_a2k.bpe --vocabulary data/keywords_train_a2k_vocab.bpe --vocabulary-threshold 1 < data/keywords_test_a2k_tokens.txt > data/keywords_test_a2k_tokens.bpe
subword-nmt apply-bpe -c data/codes_a2k.bpe --vocabulary data/abstracts_train_a2k_vocab.bpe --vocabulary-threshold 1 < data/abstracts_test_a2k_tokens.txt > data/abstracts_test_a2k_tokens.bpe

python3.6 preprocessing_a2k.py write_bpe_vocab
python3.6 preprocessing_a2k.py write_train_hdf5


