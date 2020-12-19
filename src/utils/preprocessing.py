import glob
import json

import pandas as pd


class Preprocessor:
    def __init__(self, path, lower=True, removeNonAlphaNumeric=True,
                 substituespecial=True, lemmanize=False):
        self.path = path
        self.lower = lower
        self.removeNonAlphaNumeric = removeNonAlphaNumeric
        self.substituespecial = substituespecial
        self.lemmanize = lemmanize

    def import_jsons(self):
        files = glob.glob(self.path + '*.json')
        if not files:
            raise Exception("No JSON files found!")
        return files

    def extract_text(self, file):
        with open(self.path + file, "r") as f:
            json_f = json.load(f)
            return pd.Series([review["text"] for review in json_f['reviews']])

    def extract_rating(self, file):
        with open(self.path + file, "r") as f:
            json_f = json.load(f)
            return pd.Series([review["rating"] for review in json_f['reviews']])

    def normalize():
        pass

    def prep():
        pass