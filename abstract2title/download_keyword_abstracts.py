import pymongo
import os
import pickle
import pandas as pd
import urllib
from random import shuffle
from itertools import zip_longest


def grouper(iterable, n, fillvalue=None):
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)


client = pymongo.MongoClient(os.getenv("MATSCHOLAR_STAGING_HOST"), username=os.getenv("MATSCHOLAR_STAGING_USER"),
                             password=os.getenv("MATSCHOLAR_STAGING_PASS"), authSource=os.getenv("MATSCHOLAR_STAGING_DB"))
db = client['matscholar_staging']
rel_strings = []
keyword_max = []
for i, d in enumerate(db.entries.find()):
    if i % 100000 == 0:
        print(i)
    try:
        if d['keywords'] is not None and d['abstract'] is not None:
            if isinstance(d['keywords'], list):
                keywords = d['keywords']
            else:
                keywords = d['keywords'].split().sort()
            if len(keywords) > len(keyword_max):
                keyword_max = keywords
            rel_strings.append((
                ' '.join(keywords), ' '.join(d['abstract'].split())))
    except:
        pass

print(keyword_max)
shuffle(rel_strings)

train_split = int(0.95 * len(rel_strings))

titles_train = os.path.join("data", "keywords_train_a2k.txt")
abstracts_train = os.path.join("data", "abstracts_train_a2k.txt")
titles_test = os.path.join("data", "keywords_test_a2k.txt")
abstracts_test = os.path.join("data", "abstracts_test_a2k.txt")

if not os.path.exists(os.path.dirname(titles_train)):
    os.makedirs(os.path.dirname(titles_train))

with open(titles_train, 'w') as f:
    with open(abstracts_train, 'w') as g:
        for t, a in rel_strings[:train_split]:
            f.write(t + "\n")
            g.write(a + "\n")

with open(titles_test, 'w') as f:
    with open(abstracts_test, 'w') as g:
        for t, a in rel_strings[train_split:]:
            f.write(t + "\n")
            g.write(a + "\n")
