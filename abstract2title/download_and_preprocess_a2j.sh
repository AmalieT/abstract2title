# /bin/bash

# python3.6 download_journal_abstracts.py
# python3.6 preprocessing_a2j.py write_tokenized_corpus

# subword-nmt learn-bpe -s 20000 < data/abstracts_train_a2j_tokens.txt > data/codes_a2j.bpe


# subword-nmt apply-bpe -c data/codes_a2j.bpe  < data/abstracts_train_a2j_tokens.txt > data/abstracts_train_a2j_tokens.bpe
# subword-nmt apply-bpe -c data/codes_a2j.bpe  < data/abstracts_test_a2j_tokens.txt > data/abstracts_test_a2j_tokens.bpe

# subword-nmt get-vocab < data/abstracts_train_a2j_tokens.bpe > data/abstracts_train_a2j_vocab.bpe

# python3.6 preprocessing_a2j.py write_bpe_vocab
python3.6 preprocessing_a2j.py write_train_tfrecord


