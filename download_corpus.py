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
for i, d in enumerate(db.entries.find()):
    if i % 10000 == 0:
        print(i)
    try:
        if d['title'] is not None and d['abstract'] is not None:
            rel_strings.append((
                ' '.join(d['title'].split()), ' '.join(d['abstract'].split())))
    except:
        pass

shuffle(rel_strings)

train_split = int(0.9 * len(rel_strings))

titles_train = os.path.join("data", "titles_train.txt")
abstracts_train = os.path.join("data", "abstracts_train.txt")
titles_test = os.path.join("data", "titles_test.txt")
abstracts_test = os.path.join("data", "titles_test.txt")

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
