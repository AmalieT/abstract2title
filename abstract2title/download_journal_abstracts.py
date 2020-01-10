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


min_paper_count = 10
client = pymongo.MongoClient(os.getenv("MATSCHOLAR_STAGING_HOST"), username=os.getenv("MATSCHOLAR_STAGING_USER"),
                             password=os.getenv("MATSCHOLAR_STAGING_PASS"), authSource=os.getenv("MATSCHOLAR_STAGING_DB"))
db = client['matscholar_staging']

rel_strings = []
class_dict = dict()
i_class = 0
for i, d in enumerate(db.entries.find()):
    if i % 10000 == 0:
        print(i)
    try:
        if d['journal'] is not None and d['abstract'] is not None:
            try:
                current_count = class_dict[d['journal']]
                class_dict[l] = (
                    current_count[0], current_count[1] + 1)
            except KeyError:
                class_dict[d['journal']] = (i_class, 1)
                i_class += 1
    except:
        pass

class_dict = {k: v[0]
              for k, v in class_dict.items() if v[1] >= min_paper_count}

for i, d in enumerate(db.entries.find()):
    if i % 10000 == 0:
        print(i)
    try:
        if d['journal'] is not None and d['abstract'] is not None:
            try:
                journal_class = class_dict[d['journal']]
                rel_strings.append(
                    (journal_class, ' '.join(d['abstract'].split())))
            except KeyError:
                pass
    except:
        pass


inv_class_dict = {v: k for k, v in class_dict.items()}
pickle.dump(class_dict, open(os.path.join("data", 'journal2class.pkl'), 'wb'))
pickle.dump(inv_class_dict, open(
    os.path.join("data", 'class2journal.pkl'), 'wb'))

shuffle(rel_strings)

train_split = int(0.9 * len(rel_strings))

journals_train = os.path.join("data", "journals_train_a2j.txt")
abstracts_train = os.path.join("data", "abstracts_train_a2j.txt")
journals_test = os.path.join("data", "journals_test_a2j.txt")
abstracts_test = os.path.join("data", "abstracts_test_a2j.txt")


if not os.path.exists(os.path.dirname(journals_train)):
    os.makedirs(os.path.dirname(journals_train))

with open(journals_train, 'w') as f:
    with open(abstracts_train, 'w') as g:
        for t, a in rel_strings[:train_split]:
            f.write(t + "\n")
            g.write(a + "\n")

with open(journals_test, 'w') as f:
    with open(abstracts_test, 'w') as g:
        for t, a in rel_strings[train_split:]:
            f.write(t + "\n")
            g.write(a + "\n")
