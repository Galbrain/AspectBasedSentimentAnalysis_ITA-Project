# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import json
import os
import re
from collections import Counter

import matplotlib as mlp
import numpy as np
import pandas as pd
import spacy
from matplotlib import pyplot as plt
from wordcloud import WordCloud


# load data
def jsonToSeries(path):
    """
    in:
        path: string to folder containing json files
    out:
        data: ndarray containing strings
    """
    files = os.listdir(path)
    files = [item for item in files if item.find(".gitkeep") == -1]

    text = []
    # rating = []
    for singlefile in files:
        with open(path + "/" + singlefile, "r") as f:
            json_file = json.load(f)
            reviews = json_file["reviews"]
            for review in reviews:
                if review["rating"] != {}:
                    text.append(review["text"])
            # text.append([json_file['reviews']['text']])
    return pd.Series(text)


def normalize_files(series):
    """
    in:
        series: panda series containing the text data
    out:
        series: normalized panda series
    """
    # strip beginning and end of Review
    # "Von %USERNAME% (int):
    # "Ist diese Meinung hilfreich?" "INT von INT Lesern fanden diese Meinung hilfreich. Was denkst du?"
    series = series.str.replace("\n", "")

    specialCharMap = {ord("ä"): "ae", ord("ü"): "ue", ord("ö"): "oe", ord("ß"): "ss"}
    series = series.apply(lambda x: x.translate(specialCharMap))
    series = series.str.replace(r"Von\s\w+\s+(\(\d+\))?:", "")
    series = series.str.replace(r"Ist diese Meinung hilfreich\?", "")
    series = series.str.replace(r"\d+\s\w+\s\d+(\s\w+)+\.(\s\w+)+\?", "")
    series = series.str.replace(r"[^\w\s]", "").str.lower()
    # series = series.str.replace('ae', 'ä').replace('oe', 'ö').replace('ue', 'ü')
    series = series.str.split()
    series = removeStopwords(series)

    return series


def removeStopwords(series):
    nlp = spacy.load("de_core_news_sm", disable=["tagger", "parser", "ner"])
    stopwords = nlp.Defaults.stop_words

    specialCharMap = {ord("ä"): "ae", ord("ü"): "ue", ord("ö"): "oe", ord("ß"): "ss"}
    stopwords = [word.translate(specialCharMap) for word in stopwords]

    return series.apply(
        lambda x: [word for word in x if (word not in stopwords and word != "")]
    )


def seriesToFlatArray(series):
    sentenceList = series.tolist()
    wordList = [item for sublist in sentenceList for item in sublist]

    return wordList


def mostCommonWords(series, number):
    wordList = seriesToFlatArray(series)
    counter = Counter(wordList)
    topWords = np.array(counter.most_common(number))

    return topWords


def plotCommonWords(array):
    plt.figure(figsize=(16, 9))
    plt.xticks(rotation="vertical")
    plt.bar(array[:, 0], array[:, 1].astype("int"))
    plt.savefig("src/data/pictures/most_common.png")


def plotWordCloud(series):
    wordList = seriesToFlatArray(series)
    wordcloud = WordCloud(max_words=100, max_font_size=60, collocations=False)

    spokenWords = ",".join(wordList)
    wordcloud.generate(spokenWords)
    plt.figure(figsize=(16, 9))
    mlp.rcParams["image.interpolation"] = "bilinear"
    plt.imshow(wordcloud)
    plt.savefig("src/data/pictures/word_cloud_graph.png")


def plotEntities(series):
    nlp = spacy.load("de_core_news_sm")
    reviews = seriesToFlatArray(series)[:30000]

    df_entities = []
    doc = nlp(" ".join(reviews))
    for ent in doc.ents:
        df_entities.append({"entitytext": ent.text, "entitiylabel": ent.label_})

    df_entities = pd.DataFrame(df_entities)
    df_entities_grouped = df_entities.groupby(["entitiylabel"])["entitytext"].apply(
        list
    )
    print(df_entities_grouped)


if __name__ == "__main__":
    data = jsonToSeries("src/data/raw")
    normalizedData = normalize_files(data)
    mostCommon = mostCommonWords(normalizedData, 20)
    plotCommonWords(mostCommon)
    plotWordCloud(normalizedData)
    # plotEntities(normalizedData)
