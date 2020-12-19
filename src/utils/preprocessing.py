# -*- coding: utf-8 -*-
import glob
import json

import pandas as pd


class Preprocessor:
    """
    Preprocessor class
    """

    def __init__(
        self,
        path,
        lower=True,
        removeNonAlphaNumeric=True,
        substituespecial=True,
        lemmanize=False,
    ):
        """
        Constructor for Preprocessor class

        Args:
            path (str): path to data as string
            lower (bool, optional): remove capitalization. Defaults to True.
            removeNonAlphaNumeric (bool, optional): removeNonAlphanumeric. Defaults to True.
            substituespecial (bool, optional): substitue {ä,ö,ü,ß}. Defaults to True.
            lemmanize (bool, optional): find the lemma of all words in text. Defaults to False.
        """

        self.path = path
        self.lower = lower
        self.removeNonAlphaNumeric = removeNonAlphaNumeric
        self.substituespecial = substituespecial
        self.lemmanize = lemmanize

    def find_jsons(self):
        """
        Import Json files from path

        Raises:
            Exception: No files found

        Returns:
            list(str): returns a list with filenames of JSON files
        """

        files = glob.glob(self.path + "*.json")
        if not files:
            raise Exception("No JSON files found!")
        return files

    def extract_text(self, file):
        """
        extract text from reviews in file

        Args:
            file (str): filename of the file in path

        Returns:
            pd.Series: returns Series with text from reviews
        """

        with open(self.path + file, "r") as f:
            json_f = json.load(f)
            return pd.Series([review["text"] for review in json_f["reviews"]])

    def extract_rating(self, file):
        """
        extract rating from reviews in file

        Args:
            file (str): filename of the file in path

        Returns:
            pd.Series: returns Series with ratings of the reviews
        """
        with open(self.path + file, "r") as f:
            json_f = json.load(f)
            return pd.Series([review["rating"] for review in json_f["reviews"]])

    def normalize(self):
        pass

    def prep(self):
        pass
