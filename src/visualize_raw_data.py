# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import json
import os
import re
from collections import Counter

import numpy as np
import pandas as pd
import spacy
from matplotlib import pyplot as plt


# load data
def jsonToSeries(path):
    '''
        in:
            path: string to folder containing json files
        out:
            data: ndarray containing strings
    '''
    files = os.listdir(path)
    files = [item for item in files if item.find('.gitkeep') == -1]

    text = []
    # rating = []
    for singlefile in files:
        with open(path + '/' + singlefile, 'r') as f:
            json_file = json.load(f)
            reviews = json_file['reviews']
            for review in reviews:
                if (review['rating'] != {}):
                    text.append(review['text'])
            # text.append([json_file['reviews']['text']])
    return pd.Series(text)


def normalize_files(series):
    '''
        in:
            series: panda series containing the text data
        out:
            series: normalized panda series
    '''
    # strip beginning and end of Review
    # "Von %USERNAME% (int):
    # "Ist diese Meinung hilfreich?" "INT von INT Lesern fanden diese Meinung hilfreich. Was denkst du?"
    series = series.str.replace('\n', '')

    spcial_char_map = {ord('ä'): 'ae', ord('ü'): 'ue', ord('ö'): 'oe', ord('ß'): 'ss'}
    series = series.apply(lambda x: x.translate(spcial_char_map))
    series = series.str.replace('Von\s\w+\s+(\(\d+\))?:', '')
    series = series.str.replace('Ist diese Meinung hilfreich\?', '')
    series = series.str.replace('\d+\s\w+\s\d+(\s\w+)+\.(\s\w+)+\?', '')
    series = series.str.replace('[^\w\s]', '').str.lower()
    series = series.str.split()
    series = removeStopwords(series)

    return series


def removeStopwords(series):
    nlp = spacy.load("de_core_news_sm", disable=["tagger", "parser", "ner"])
    stopwords = nlp.Defaults.stop_words
    return series.apply(
        lambda x: [word for word in x if(word not in stopwords and word != '')])


def mostCommonWords(series, number):
    sentenceList = series.tolist()
    wordList = [item for sublist in sentenceList for item in sublist]

    counter = Counter(wordList)
    top_words = np.array(counter.most_common(number))
    return top_words


def plotCommonWords(array):
    plt.figure(figsize=(16, 9))
    plt.xticks(rotation='vertical')
    plt.bar(array[:, 0], array[:, 1].astype('int'))
    plt.savefig('src/data/pictures/most_common.png')


if __name__ == "__main__":
    data = jsonToSeries("src/data/raw")
    normalizedData = normalize_files(data)
    mostCommon = mostCommonWords(normalizedData, 20)
    plotCommonWords(mostCommon)
