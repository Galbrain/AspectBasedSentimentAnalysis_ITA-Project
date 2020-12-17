import glob
import json
import re
from os import path

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
    files = glob.glob(path + '/*.json', recursive=True)

    text = []
    # rating = []
    for singlefile in files:
        with open(singlefile, 'r') as f:
            json_file = json.load(f)

            text.append([
                json_file['text']
            ])
            # rating.append([
            #    json_file['rating']
            # ])

    return pd.Series(data)


def normalize_files(series):
    # strip beginning and end of Review
    # "Von %USERNAME% (int):
    # "Ist diese Meinung hilfreich?" "INT von INT Lesern fanden diese Meinung hilfreich. Was denkst du?"

    series = series.apply(lambda x: x.encode('utf-8').decode('unicode-escape'))
    series = series.str.replace('[^\w\s]', '').str.lower()

    return normalizedArray


if __name__ == "__main__":
    print(jsonToArray("./data"))
