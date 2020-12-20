# -*- coding: utf-8 -*-
import glob
import json
import re

import pandas as pd
import spacy


class Preprocessor:
    """
    Preprocessor class
    """

    def __init__(
        self,
        path,
        lower=True,
        rmnonalphanumeric=True,
        substituespecial=True,
        lemmanize=False,
        rmstopwords=True,
        rmdefault=True,
    ):
        """
        Constructor for Preprocessor class

        Args:
            path (str): path to data as string
            lower (bool, optional): remove capitalization. Defaults to True.
            rmnonalphanumeric (bool, optional): removeNonAlphanumeric. Defaults to True.
            substituespecial (bool, optional): substitue {ä,ö,ü,ß}. Defaults to True.
            lemmanize (bool, optional): find the lemma of all words in text. Defaults to False.
            rmstopwords (bool, optional): rm stopwords from spacey stopword list. Defaults to True.
            rmdefault (bool, optional): rm default phrase from the text. Defaults to True.
        """

        self.path = path
        self.lower = lower
        self.rmnonalphanumeric = rmnonalphanumeric
        self.substituespecial = substituespecial
        self.lemmanize = lemmanize
        self.rmstopwords = rmstopwords
        self.rmdefault = rmdefault
        self.stopwords = {}
        self.substitutedict = {}

    def find_jsons(self):
        """
        find JSON files from path

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

        with open(file, "r") as f:
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

        with open(file, "r") as f:
            json_f = json.load(f)
            return pd.Series([review["rating"] for review in json_f["reviews"]])

    def removeCapitalization(self, series):
        """
        remove capitalization from the series

        Args:
            series (pd.Series): Series containing all text from reviews

        Returns:
            pd.Series: Series with all lower letters
        """

        series = series.str.lower()
        return series

    def rmNonAlphaNumeric(self, series):
        """
        remove all non alpha numerica characters from the text in series

        Args:
            series (pd.Series): Series containing all text from reviews

        Returns:
            pd.Series: Series without non alpha numeric text
        """

        return series.str.replace(r"[^\w\s]", "")

    def removeString(self, series, regstring):
        """
        remove strings with schema

        Args:
            series (pd.Series): Series containing review text
            regstring (str, optional): string to be removed.

        Returns:
            pd.Series: Series with strings Removed
        """

        return series.str.replace(regstring, "")

    def removeDefaultStrings(self, series):
        """
        remove default string: Von %USER% (1) :, Ist diese Meinung hilfreich?, INT von INT Lesern fand diese Meinung hilfreich

        Args:
            series (pd.Series): Series containing text

        Returns:
            pd.Series: Series without the most common phrased generated by the website
        """

        series = self.removeString(series, "\n")
        series = self.removeString(series, r"[vV]on\s\w+\s+(\(\d+\))?:")
        series = self.removeString(series, r"[iI]st diese [mM]einung hilfreich(\?)?")
        series = self.removeString(series, r"\d+\s\w+\s\d+(\s\w+)+\.(\s\w+)+\?")
        return series

    def substitueSpecial(
        self,
        series,
        dict={ord("ä"): "ae", ord("ü"): "ue", ord("ö"): "oe", ord("ß"): "ss"},
    ):
        """
        substitue the special characters for text and stopwords with their non utf-8 counterparts

        Args:
            series (pd.Series): Series containing the text from reviews
            dict (dict, optional): dictionary of the characters to substitute. Defaults to {ord("ä"): "ae", ord("ü"): "ue", ord("ö"): "oe", ord("ß"): "ss"}.

        Returns:
            pd.Series: Series containing text with subsituted special characters
        """

        self.substitutedict = dict
        return series.apply(lambda x: x.translate(self.substitutedict))

    def tokenize(self, series):
        """
        tokenize the series

        Args:
            series (pd.Series): Series containing text

        Returns:
            pd.Series: Series containing array of tokens
        """

        return series.str.split()

    def loadSpacyModel(
        self, model="de_core_news_sm", disableList=["tagger", "parser", "ner"]
    ):
        """
        load the spacy model with required modes

        Args:
            model (str, optional): name of the mode. Defaults to "de_core_news_sm".
            disableList (list, optional): list of things to be disabled. Defaults to ["tagger", "parser", "ner"].

        Returns:
            Language: Language object with the loaded model
        """

        try:
            nlp = spacy.load(model, disable=disableList)
        except OSError:
            spacy.cli.download(model)
            return self.loadSpacyModel(model, disableList)
        return nlp

    def loadStopwords(self, nlp):
        """
        load Stopwords from spacy

        Args:
            nlp (Language): Language object with loaded model
        """

        self.stopwords = nlp.Defaults.stop_words

    def removeStopwords(self, series, nlp):
        """
        removes the stopwords from tokenized series

        Args:
            series (pd.Series): Series containing tokenized text
            nlp (Language): Language object with loaded model

        Returns:
            pd.Series: tokenized Series without stopwords
        """

        self.loadStopwords(nlp)

        if self.substituespecial:
            self.stopwords = {
                word.translate(self.substitutedict) for word in self.stopwords
            }

        return series.apply(
            lambda x: [
                word for word in x if (word not in self.stopwords and word != "")
            ]
        )

    def prep(self):
        """
        prepares the data with the configuations given at initialization

        Returns:
            pd.Series: tokenized Series
        """

        text = pd.Series(dtype=str)
        files = self.find_jsons()

        for file in files:
            text = text.append(self.extract_text(file), ignore_index=True)

        if self.substituespecial:
            text = self.substitueSpecial(text)

        if self.rmnonalphanumeric:
            text = self.rmNonAlphaNumeric(text)

        if self.lower:
            text = self.removeCapitalization(text)

        if self.rmdefault:
            text = self.removeDefaultStrings(text)

        if self.rmstopwords:
            nlp = self.loadSpacyModel()
            text_tokenized = self.tokenize(text)
            return self.removeStopwords(text_tokenized, nlp)
        else:
            return self.tokenize(text)
