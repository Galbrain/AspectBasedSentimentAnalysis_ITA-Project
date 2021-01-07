# -*- coding: utf-8 -*-
import os
import re

import numpy as np
import pandas as pd


class AspectAnnotator:
    def __init__(
        self,
        path: str = "src/data/",
        data: pd.DataFrame = None,
        keyWords: dict = {
            "Grafik": ["grafik", "optik"],
            "Sound": ["sound", "klang", "ton", "akustik"],
            "Steuerung": ["steuerung", "bedienung"],
            "Atmosphäre": ["atmosphäre", "stimmung"],
        },
    ) -> None:

        self.path = path
        self.data = data
        self.keyWords = keyWords

    def loadCSV(self, filename: str = "data_preprocessed.csv"):
        self.data = pd.read_csv(self.path + filename)

    def findAspects(self, rowDf):
        aspects = {}
        for aspect in self.keyWords:
            compare = np.array(self.keyWords[aspect])
            aspects[aspect] = np.flatnonzero(
                np.core.char.find(np.array(rowDf["tokens"]), compare) != -1
            )

    def annotate(self):
        self.data.apply(lambda x: self.findAspects(x), axis=1)
