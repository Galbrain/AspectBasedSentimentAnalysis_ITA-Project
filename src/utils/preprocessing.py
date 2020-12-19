# -*- coding: utf-8 -*-
import glob
import json

import pandas as pd


class Preprocessor:
    def __init__(
        self,
        path,
        lower=True,
        removeNonAlphaNumeric=True,
        substituespecial=True,
        lemmanize=False,
    ):
        """Constructor for Preprocessor

        :param path: Path to raw data
        :type path: str
        :param lower: remove capital letters, defaults to True
        :type lower: bool, optional
        :param removeNonAlphaNumeric: remove all alpha numeric characters, defaults to True
        :type removeNonAlphaNumeric: bool, optional
        :param substituespecial: subsitute ä ö ü ß, defaults to True
        :type substituespecial: bool, optional
        :param lemmanize: create the lemma of the words in the data, defaults to False
        :type lemmanize: bool, optional
        """

        self.path = path
        self.lower = lower
        self.removeNonAlphaNumeric = removeNonAlphaNumeric
        self.substituespecial = substituespecial
        self.lemmanize = lemmanize

    def import_jsons(self):
        """Import JSONs from path

        :raises Exception: No files found
        :return: List of filesnames
        :rtype: [str]
        """

        files = glob.glob(self.path + "*.json")
        if not files:
            raise Exception("No JSON files found!")
        return files

    def extract_text(self, file):
        """extract text from reviews in file

        :param file: name of jsonfile
        :type file: str
        :return: panda series containing the text of the reviews
        :rtype: pd.Series
        """

        with open(self.path + file, "r") as f:
            json_f = json.load(f)
            return pd.Series([review["text"] for review in json_f["reviews"]])

    def extract_rating(self, file):
        """extract ratings from reviews in file

        :param file: name of jsonfile
        :type file: str
        :return: panda series containing the ratings of the reviews
        :rtype: pd.Series
        """
        with open(self.path + file, "r") as f:
            json_f = json.load(f)
            return pd.Series([review["rating"] for review in json_f["reviews"]])

    def normalize():
        pass

    def prep():
        pass
