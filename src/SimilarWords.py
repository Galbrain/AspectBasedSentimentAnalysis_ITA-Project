import itertools
import os
import re

import pandas as pd
import spacy
from gensim.models import KeyedVectors, Word2Vec
from tqdm import tqdm
from utils.preprocessing import Preprocessor


def train_w2v(reviews):

    def hash(astring):
        return ord(astring[0])

    w2v = Word2Vec(
        min_count=2, size=100, alpha=0.03, negative=20, window=3, min_alpha=0.0001,
        sample=0.00006, hashfxn=hash, workers=1)  # your code ###
    w2v.build_vocab(reviews)
    w2v.train(reviews, total_examples=w2v.corpus_count, epochs=100)
    w2v.init_sims()
    w2v.wv.save_word2vec_format('src/data/w2v_model.bin', binary=True)
    return w2v


def normalize_text():

    preprocessor = Preprocessor()
    preprocessor.loadCSV()
    preprocessor.prep()
    tokens = preprocessor.data['tokens'].tolist()
    tokens = list(itertools.chain(*tokens))
    tokens = list(itertools.chain(*tokens))
    return tokens


def get_most_similar(aspect):

    w2v = None
    if 'w2v_model.bin' in os.listdir('src/data'):
        w2v = KeyedVectors.load_word2vec_format(
            "src/data/w2v_model.bin", binary=True, unicode_errors='ignore')
    else:
        reviews = normalize_text()
        w2v = train_w2v(reviews)

    try:
        most_similar = w2v.wv.most_similar(aspect)
        print('Most similar words for', aspect, ':', most_similar)
    except KeyError:
        print('Key', aspect, 'not found')
