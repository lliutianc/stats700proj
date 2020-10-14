import re
import csv

import nltk
from nltk.stem import *
try:
    from nltk.corpus import stopwords
except:
    nltk.download('stopwords')
    from nltk.corpus import stopwords

import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset

TRAIN_FILE = 'dataset/Train.csv'
VALID_FILE = 'dataset/Valid.csv'
TEST_FILE = 'dataset/Test.csv'

stemmer = PorterStemmer()
STOPS = set(stemmer.stem(w) for w in stopwords.words('english'))


def is_stopword(word):
    STOP_PAT = re.compile('^[a-zA-Z]{2,}$')
    return not STOP_PAT.match(word) and word not in STOPS


def data_file(data):
    assert data in ['train', 'valid', 'test'], f'invalid dataset: {data}'
    if data == 'train':
        return TRAIN_FILE
    if data == 'valid':
        return VALID_FILE
    if data == 'test':
        return TEST_FILE


class IMDBDataset(Dataset):
    def __init__(self, data, data_limit=None, balanced_limit=False):
        neg, pos = [], []

        with open(data_file(data), 'r', encoding="utf8") as file:   
            reader = csv.reader(file)
            next(reader)
            for idx, line in enumerate(reader):
                words = [stemmer.stem(word) for word in line[0].split() if not is_stopword(word)]
                if line[1] == '1':
                    if not balanced_limit:
                        pos.append(words)
                    else:
                        if len(pos) < (data_limit // 2):
                            pos.append(words)
                else:
                    if not balanced_limit:
                        neg.append(words)
                    else:
                        if len(neg) < (data_limit // 2):
                            neg.append(words)

                if data_limit is not None:
                    if (idx + 1) >= data_limit:
                        break

        self.pos = pos
        self.neg = neg

        self.x = self.pos + self.neg
        self.y = [1 for _ in range(len(pos))] + [0 for _ in range(len(neg))]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i]

    def shuffle(self):
        order = np.arange(len(self))
        np.random.shuffle(order)
        self.x = self.x[order]
        self.y = self.y[order]


def load_LDA_data(model, dataset, include_llk=False):
    lda_x = []
    llk_x = []
    for x, _ in dataset:
        x = model.make_doc(x)
        x, llk = model.infer(x)
        lda_x.append(x)
        llk_x.append(llk)

    lda_x = np.array(lda_x)
    llk_x = np.array(llk_x)
    lda_y = np.array(dataset.y)

    if include_llk is False:
        return lda_x, lda_y
    else:
        return lda_x, llk_x, lda_y


def load_LDA_data_batch(model, dataset, include_llk=False):
    docs = [model.make_doc(x) for (x, _) in dataset]
    lda_x, llk = model.infer(docs)

    lda_x = np.array(lda_x)
    llk = np.array(llk)
    lda_y = np.array(dataset.y)

    if include_llk is False:
        return lda_x, lda_y
    else:
        return lda_x, llk, lda_y