import glob
import json


class Preprocessor:
    def __init__(self, path: str, lower=True, removeNonAlphaNumeric=True,
                 substituespecial=True, lemmanize=False):
        self.path = path
        self.lower = lower
        self.removeNonAlphaNumeric = removeNonAlphaNumeric
        self.substituespecial = substituespecial
        self.lemmanize = lemmanize

    def import_jsons():
        files = glob.glob(self.path + '*.json')
        return files

    def extract_text(file):
        with open(self.path + file, "r") as f:
            json_f = json.load(f)
            return pd.Series([review["text"] for review in json_f['reviews']])

    def extract_rating(file):
        with open(self.path + file, "r") as f:
            json_f = json.load(f)
            return pd.Series([review["rating"] for review in json_f['reviews']])

    def normalize()

    def prep():
