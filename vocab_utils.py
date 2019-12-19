
from gensim.utils import to_unicode, deaccent
from chemdataextractor.doc import Paragraph
import string
from pymatgen import Composition
import re
import collections
import sys
import logging
from keras.utils.io_utils import HDF5Matrix


PAT_ALPHABETIC = re.compile(r'[a-zA-Z]+[\w\-()]*[\w\-]+', re.UNICODE)


def custom_tokenize(text, lowercase=False, deacc=False, encoding='utf8', errors="strict", to_lower=False, lower=False, cde=True):
    text = to_unicode(text, encoding, errors=errors)
    lowercase = lowercase or to_lower or lower
    if lowercase:
        text = text.lower()
    if deacc:
        text = deaccent(text)
    if cde:
        text = " ".join(text.split())
        cde_p = Paragraph(text)
        tokens = cde_p.tokens
        toks = []
        for sentence in tokens:
            toks.append([])
            for tok in sentence:
                if tok.text not in string.punctuation:
                    yield tok.text
    else:
        for match in PAT_ALPHABETIC.finditer(text):
            yield match.group()


def tokenize(content, token_min_len=2, token_max_len=80, lower=False):
    return [
        to_unicode(token) for token in custom_tokenize(content, lower=lower, errors='ignore')
        if token_min_len <= len(token) <= token_max_len and not token.startswith('_')
    ]


def tokenize_corpus(corpusFile, savefile):
    with open(savefile, 'w') as g:
        with open(corpusFile, 'r') as f:
            for i, l in enumerate(f):
                if i % 10000 == 0:
                    sys.stdout.write("Processing abstract: %d   \r" % (i))
                    sys.stdout.flush()
                abstract = tokenize(l)
                g.write(" ".join(abstract) + "\n")

    return None


def to_index(text, word2index, maxlen):
    tokenized = "<BOS> {}".format(text)
    text = tokenize(text)[:maxlen - 1] + ["<EOS>"]

    def index_or_unk(token):
        try:
            return word2index[token]
        except KeyError:
            return word2index["<UNK>"]

    text = [index_or_unk(t) for t in text]

    return text


def from_index(tokens, index2word):
    return " ".join([index2word[t] for t in tokens])


def bpe_to_index(text, word2index, maxlen):
    text_tokens = ["<BOS>"] + text.split()[:maxlen - 2] + ["<EOS>"]

    text_tokens = [word2index[t] for t in text_tokens]
    return text_tokens


def create_bpe_vocab(titleVocabFile, abstractsVocabFile):
    word2index = dict()
    counter = collections.Counter()

    with open(abstractsVocabFile, 'r') as f:
        for i, l in enumerate(f):
            if i % 10000 == 0:
                sys.stdout.write("Processing abstract: %d   \r" % (i))
                sys.stdout.flush()
            counter.update(l.split())

    with open(titleVocabFile, 'r') as f:
        for i, l in enumerate(f):
            if i % 10000 == 0:
                sys.stdout.write("Processing title: %d   \r" % (i))
                sys.stdout.flush()
            counter.update(l.split())

    word2index["<BOS>"] = 0
    word2index["<EOS>"] = 1

    for i, l in enumerate(counter.most_common()):
        word, _ = l
        word2index[word] = i + 2

    index2word = {v: k for k, v in word2index.items()}

    return word2index, index2word
