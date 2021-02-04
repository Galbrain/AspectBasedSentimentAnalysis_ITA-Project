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
        self.df = pd.DataFrame(
            columns=["reviewnumber", "word_found", "word_idx", "aspect"]
        )

    def loadCSV(self, filename: str = "data_preprocessed.csv") -> None:
        """
        load CSV from the given filename

        Args:
            filename (str, optional): String of path to the preprocessed data. Defaults to "data_preprocessed.csv".
        """
        self.data = pd.read_csv(self.path + filename)
        self.data["tokens"] = (
            self.data["tokens"]
            .str.split(",", expand=True)
            .replace(r"[\[\]]", "", regex=True)
            .astype(str)
            .values.tolist()
        )

    def findAspects(self, rowDf: pd.Series) -> None:
        """
        function to be vectorized for the dataset

        Args:
            rowDf (pd.Series): Row of a dataframe containing "index and tokens"
        """
        aspects = {}
        for aspect in self.keyWords:
            compare = self.keyWords[aspect]

            for i, elem in enumerate(rowDf["tokens"]):
                for e in compare:
                    if e in elem.lower():
                        aspects["reviewnumber"] = rowDf.name
                        aspects["word_found"] = elem
                        aspects["word_idx"] = i
                        aspects["aspect"] = aspect
                        self.df = self.df.append(aspects, ignore_index=True)

    def annotate(self) -> None:
        """
        function to call the "findAspects()" function for every row
        """
        self.data.apply(lambda x: self.findAspects(x), axis=1)

    def saveCSV(self, filename: str = "data_aspects_tokens.csv") -> None:
        """
        Save csv that contains the data for the annotation

        Args:
            filename (str, optional): path to file. Defaults to "data_aspects_tokens.csv".
        """
        self.df.to_csv(self.path + filename, index=False)
