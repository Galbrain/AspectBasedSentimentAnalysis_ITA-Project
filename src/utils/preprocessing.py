# -*- coding: utf-8 -*-
import glob
import json

import pandas as pd
import spacy
from nltk.tokenize import sent_tokenize, word_tokenize
from tqdm import tqdm


class Preprocessor:
    """
    Preprocessor class
    """

    def __init__(
        self,
        path: str = "src/data/",
        lower: bool = True,
        rmnonalphanumeric: bool = True,
        substituespecial: bool = False,
        lemmatize: bool = False,
        rmstopwords: bool = True,
        rmdefault: bool = True,
    ):
        """
        Constructor for Preprocessor class

        Args:
            path (str): path to data as string
            lower (bool, optional): remove capitalization. Defaults to True.
            rmnonalphanumeric (bool, optional): removeNonAlphanumeric. Defaults to True.
            substituespecial (bool, optional): substitue {ä,ö,ü,ß}. Defaults to True.
            lemmatize (bool, optional): find the lemma of all words in text. Defaults to False.
            rmstopwords (bool, optional): rm stopwords from spacey stopword list. Defaults to True.
            rmdefault (bool, optional): rm default phrase from the text. Defaults to True.
        """

        self.path = path
        self.lower = lower
        self.rmnonalphanumeric = rmnonalphanumeric
        self.substituespecial = substituespecial
        self.rmstopwords = rmstopwords
        self.rmdefault = rmdefault
        self.stopwords = None
        self.substitutedict = {}
        self.nlp = None
        self.normalizeColumn = "text_normalized"
        self.data = None

    def loadCSV(self, filename: str = "data_raw.csv"):
        """
        read csv of self.path + filename, drop all invalid entries and create a normalized text column

        Args:
            filename (str, optional): Defaults to "data_raw.csv".
        """
        df_DataRaw = pd.read_csv(self.path + filename)
        df_DataRaw.dropna(inplace=True)
        df_DataRaw[self.normalizeColumn] = df_DataRaw["review_text_raw"]
        self.data = df_DataRaw

    def saveCSV(self, filename: str = "data_preprocessed.csv"):
        """
        safe the data as csv

        Args:
            filename (str, optional): Defaults to "data_preprocessed.csv".
        """
        tqdm.pandas(desc="JSON dumping tokens..")
        self.data["tokens"] = self.data["tokens"].progress_apply(
            lambda x: json.dumps(x)
        )
        self.data.to_csv(self.path + filename, index=False)

    def splitSentences(self) -> None:
        tqdm.pandas(desc="Splitting Sentences")
        self.data["tokens"] = self.data[self.normalizeColumn].progress_apply(
            lambda x: sent_tokenize(x, language="german")
        )

    def removeCapitalization(self) -> bool:
        """
        remove capitalization from self.data[self.normalizeColumn]

        Returns:
            bool: Successful execution of command
        """
        self.data[self.normalizeColumn] = self.data[self.normalizeColumn].str.lower()
        return True

    def rmNonAlphaNumeric(self) -> bool:
        """
        removes the non-alpha-numeric characters from self.data['text_normlaized']

        Returns:
            bool: Sucessful execution of command
        """

        self.data[self.normalizeColumn] = self.data[self.normalizeColumn].str.replace(
            r"[^\w\s]", " ", regex=True
        )

    def removeString(self, regstring: str) -> bool:
        """
        remove strings with schema

        Args:
            regstring (str, optional): string to be removed.

        Returns:
            bool: Successful execution of command
        """

        self.data[self.normalizeColumn] = self.data[self.normalizeColumn].str.replace(
            regstring, " ", regex=True
        )

    def removeDefaultStrings(self) -> bool:
        """
        remove default string: Von %USER% (INT) :, Ist diese Meinung hilfreich?, INT von INT Lesern fand diese Meinung hilfreich

        Returns:
            bool: Successful exectuion of command
        """
        self.removeString("\n")
        self.removeString(r"[vV]on\s[\w\d-]+\s+(\(\d+\))?:")
        self.removeString(r"[iI]st diese [mM]einung hilfreich(\?)?")
        self.removeString(r"\d+\s\w+\s\d+(\s\w+)+\.(\s\w+)+\?")
        return bool

    def substitueSpecial(
        self,
        transldict: dict = {
            ord("ä"): "ae",
            ord("ü"): "ue",
            ord("ö"): "oe",
            ord("ß"): "ss",
        },
    ) -> bool:
        """
        substitue the special characters for text and stopwords with their non utf-8 counterparts

        Args:
            dict (dict, optional): dictionary of the characters to substitute. Defaults to {ord("ä"): "ae", ord("ü"): "ue", ord("ö"): "oe", ord("ß"): "ss"}.

        Returns:
            bool: Sucessfull execution of command
        """

        self.substitutedict = transldict
        self.data[self.normalizeColumn] = self.data[self.normalizeColumn].apply(
            lambda x: x.translate(self.substitutedict)
        )

    def tokenize(self) -> bool:
        """
        tokenize the series

        Returns:
            bool: Sucessful execution of command
        """
        self.splitSentences()
        tqdm.pandas(desc="Creating Word Tokens")
        self.data["tokens"] = self.data["tokens"].progress_apply(
            lambda x: [word_tokenize(i, language="german") for i in x]
        )

    def loadSpacyModel(
        self,
        model: str = "de_core_news_sm",
        disableList: list[str] = ["tagger", "parser", "ner"],
    ) -> bool:
        """
        load the spacy model with required modes

        Args:
            model (str, optional): name of the mode. Defaults to "de_core_news_sm".
            disableList (list[str], optional): list of things to be disabled. Defaults to ["tagger", "parser", "ner"].
        """
        try:
            self.nlp = spacy.load(model, disable=disableList)
            return True
        except OSError:
            print("Model not found. Attempting to download..")
            try:
                spacy.cli.download(model)
            except Exception as e:
                print(e)
                return False
            self.nlp = spacy.load(model, disable=disableList)
            return True

    def loadStopwords(self) -> bool:
        """
        load Stopwords from spacy

        Return:
            bool: whenver the function completed successfully
        """
        if not self.nlp:
            if not self.loadSpacyModel():
                print("Skipping. Unable to load SpacyModel")
                return False

        self.stopwords = self.nlp.Defaults.stop_words
        return True

    def removeStopwords(self) -> bool:
        """
        removes the stopwords from tokenized series

        Returns:
            bool: Sucessful execution of command
        """
        if self.stopwords is not None:
            return True

        if not self.loadStopwords():
            print("Skipping. Unable to load Stopwords!")

        if self.substituespecial:
            self.stopwords = {
                word.translate(self.substitutedict) for word in self.stopwords
            }

        self.data["tokens"] = self.data["tokens"].apply(
            lambda s: [
                [word for word in sentence if word not in self.stopwords and word != ""]
                for sentence in s
            ]
        )

    def prep(self) -> bool:
        """
        prepares the data with the configuations given at initialization

        Returns:
            pd.Series: tokenized Series
        """
        if self.data is None or self.data.empty:
            self.loadCSV()

        if self.rmdefault:
            self.removeDefaultStrings()

        if self.substituespecial:
            self.substitueSpecial()

        if self.rmnonalphanumeric:
            self.rmNonAlphaNumeric()

        if self.lower:
            self.removeCapitalization()

        if self.rmstopwords:
            self.tokenize()
            self.removeStopwords()
        else:
            self.tokenize()

        return True
