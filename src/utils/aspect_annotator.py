# -*- coding: utf-8 -*-
import json
import os
import re

import numpy as np
import pandas as pd
from tqdm import tqdm


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
        if os.path.exists("src/data/aspectDict.json"):
            with open("src/data/aspectDict.json") as f:
                self.keyWords = json.load(f)
        self.df = pd.DataFrame(
            columns=["reviewnumber", "word_found",
                     "sent_idx", "word_idx", "aspect"]
        )

    def loadCSV(self, filename: str = "data_preprocessed.csv") -> None:
        """
        load CSV from the given filename

        Args:
            filename (str, optional): String of path to the preprocessed data. Defaults to "data_preprocessed.csv".
        """
        self.data = pd.read_csv(self.path + filename, lineterminator="\n")
        tqdm.pandas(desc="Loading Tokens..")
        self.data["tokens"] = self.data["tokens"].progress_apply(
            lambda x: json.loads(x)
        )

    def findAspects(self, rowDf: pd.DataFrame) -> None:
        """
        function to be vectorized for the dataset

        Args:
            rowDf (pd.Dataframe): Row of a dataframe containing "index (rowDF.name)" and "tokens"
        """
        aspects = {}
        for aspect in self.keyWords:
            compare = self.keyWords[aspect]

            for i, sent in enumerate(rowDf["tokens"]):
                for j, word in enumerate(sent):
                    for e in compare:
                        if e in word.lower():
                            aspects["reviewnumber"] = rowDf.name
                            aspects["word_found"] = word
                            aspects["sent_idx"] = i
                            aspects["word_idx"] = j
                            aspects["aspect"] = aspect
                            self.df = self.df.append(
                                aspects, ignore_index=True)

    def annotate(self) -> None:
        """
        function to call the "findAspects()" function for every row
        """
        tqdm.pandas(desc="Finding Aspects!")
        self.data.progress_apply(lambda x: self.findAspects(x), axis=1)

    def saveCSV(self, filename: str = "data_aspects_tokens.csv") -> None:
        """
        Save csv that contains the data for the annotation

        Args:
            filename (str, optional): path to file. Defaults to "data_aspects_tokens.csv".
        """
        self.df.to_csv(self.path + filename, index=False)
