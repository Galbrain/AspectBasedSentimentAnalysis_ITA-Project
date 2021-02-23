import json
import os
from enum import Enum

import numpy as NP
import pandas as PD
import requests
import spacy
from germalemma import GermaLemma
from spacy import displacy
from tqdm import tqdm


class ChildType(Enum):
    DESCRIPTOR = 0
    INTENSIFIER = 1


class SentimentDetector:
    def __init__(self, path: str = "src/data/", windowSize=5) -> None:
        self.path = path
        self.windowSize = windowSize

        self.df_aspect_tokens = None
        self.df_preprocessed = None
        self.df_lexicon = None

        self.lemmatizer = GermaLemma()

    def downloadLexicon(
        self,
        filename: str = "sentiment_lexicon.csv",
        url: str = "https://raw.githubusercontent.com/sebastiansauer/pradadata/master/data-raw/germanlex.csv",
        chunk_size: int = 1024,
    ) -> None:
        """
        Download sentiment lexicon.

        Args:
            filename (str, optional):  Defaults to "sentimentLexicon.csv".
            url (str, optional):  Defaults to "https://raw.githubusercontent.com/sebastiansauer/pradadata/master/data-raw/germanlex.csv".
            chunk_size (int, optional): Defines chunk size for downloads of bigger files. Defaults to 128.
        """
        r = requests.get(url, stream=True)

        file_size = int(r.headers.get("Content-Length", None))
        num_bars = NP.ceil(file_size / (chunk_size))

        downloadProgress = tqdm(
            total=num_bars, desc="Downloading Lexicon...", unit="B", unit_scale=True
        )

        with open(self.path + filename, "wb") as fd:
            for chunk in r.iter_content(chunk_size=chunk_size):
                downloadProgress.update(len(chunk))
                fd.write(chunk)
        downloadProgress.close()

    def loadCSVs(
        self,
        tokenFilename: str = "data_aspects_tokens.csv",
        preprocessedFilename: str = "data_preprocessed.csv",
        lexiconFilename: str = "sentiment_lexicon.csv",
    ) -> bool:
        """
        load all necessary CSV for execution of the detector and set indices as appropriate

        Args:
            tokenFilename (str, optional): Defaults to "data_aspects_tokens.csv".
            preprocessedFilename (str, optional): Defaults to "data_preprocessed.csv".
            lexiconFilename (str, optional): Defaults to "sentiment_lexicon.csv".

        Returns:
            bool: successful execution
        """
        try:
            if self.df_aspect_tokens is None or self.df_aspect_tokens.empty:
                self.df_aspect_tokens = PD.read_csv(self.path + tokenFilename)

                self.df_aspect_tokens["polarity_strength"] = PD.NaT
                self.df_aspect_tokens["polarity_strength"].fillna(
                    {i: [] for i in self.df_aspect_tokens.index}, inplace=True
                )

                self.df_aspect_tokens["sentiment_words"] = PD.NaT
                self.df_aspect_tokens["sentiment_words"].fillna(
                    {i: [] for i in self.df_aspect_tokens.index}, inplace=True
                )

                self.df_aspect_tokens["intensifier_words"] = PD.NaT
                self.df_aspect_tokens["intensifier_words"].fillna(
                    {i: [] for i in self.df_aspect_tokens.index}, inplace=True
                )

                self.df_aspect_tokens["word_found"] = self.df_aspect_tokens[
                    "word_found"
                ].str.replace(r"[^\w]*", "", regex=True)

                # TODO remove after debugging
                # self.df_aspect_tokens = self.df_aspect_tokens[:100]

            if self.df_preprocessed is None or self.df_preprocessed.empty:
                self.df_preprocessed = PD.read_csv(self.path + preprocessedFilename)

                # pandas read_csv does not read arrays correctly so we need to adjust those
                tqdm.pandas(desc="Applying Datatype Transformations....")
                self.df_preprocessed["tokens"] = self.df_preprocessed[
                    "tokens"
                ].progress_apply(lambda x: json.loads(x))

            if self.df_lexicon is None or self.df_lexicon.empty:
                if not os.path.exists(self.path + lexiconFilename):
                    self.downloadLexicon()

                self.df_lexicon = PD.read_csv(self.path + lexiconFilename)
                self.df_lexicon.drop_duplicates(
                    subset=["word", "qualifier"], inplace=True
                )
                self.df_lexicon.set_index("word", inplace=True)
                self.df_lexicon.drop("%%")

            return True
        except IOError as e:
            print(e)
            return False

    def loadSpacyModel(
        self,
        model: str = "de_core_news_lg",
        disableList: list[str] = ["ner", "textcat"],
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

    def checkValidChild(self, child, childType: ChildType) -> bool:
        if childType == ChildType.DESCRIPTOR:
            if (child.tag_ == "ADJA" and child.pos_ == "ADJ") or (child.pos_ == "ADV" and child.tag_ == "ADJD"):
                return True
            return False
        elif childType == ChildType.INTENSIFIER:
            if child.pos_ == "ADJ" or child.pos_ == "ADV":
                return True
            return False
        else:
            print("Wrong childType.")
            return False

    def checkPolarityAdjective(self, child, rowIdx) -> float:
        """
        check if the given word has an entry in the sentiment lexicon and return given polarity strength

        Args:
            child (spacy.Token): tokenized word with tagged 'pos_' and 'text'

        Returns:
            pol_strength (float): polarity_strength of given word found in sentiment lexicon
        """
        child_normalized = child.text.replace(r"[^\w]*", "")

        lexEntry = self.checkLexicon(child_normalized)

        if lexEntry is None:
            lexEntry = self.checkLexicon(child_normalized.lower())

        if lexEntry is None:
            lemma = self.lemmatizer.find_lemma(child_normalized, child.pos_)
            lexEntry = self.checkLexicon(lemma)

        if lexEntry is None:
            return 1

        if type(lexEntry["qualifier"]) == str:
            pol_strength = lexEntry["polarity_strength"]
            if lexEntry["qualifier"] == "NEG":
                return -pol_strength
            return pol_strength
        else:
            for i, qualifier in enumerate(lexEntry["qualifier"].values):
                if qualifier == "POS":
                    return lexEntry["polarity_strength"][i]
                if qualifier == "NEG":
                    return -lexEntry["polarity_strength"][i]
            return 1

    def checkLexicon(self, word) -> PD.Series:
        """
        Check for valid lexicon entries return None if not found

        Args:
            word (str): word to be use as key

        Returns:
            PD.Series: Series that is found for the given key or None
        """
        try:
            return self.df_lexicon.loc[word]
        except KeyError:
            return None

    def checkForIntensifier(self, child, rowIdx) -> float:
        """
        For a given spacy.Token (child) check if any of the children is an intensifier and if so, return their polarity_strength

        Args:
            child (spacy.Token): tokenized word with tagged 'pos_' and 'text'

        Returns:
            polarity_multiplier (float): polarity_multiplier of found intensifier word
        """
        child_normalized = child.text.replace(r"[^\w]*", "")
        # catch words that are not in the sentiment lexicon

        lexEntry = self.checkLexicon(child_normalized)

        if lexEntry is None:
            lexEntry = self.checkLexicon(child_normalized.lower())

        if lexEntry is None:
            lemma = self.lemmatizer.find_lemma(child_normalized, child.pos_)
            lexEntry = self.checkLexicon(lemma)

        if lexEntry is None:
            return 1

        if type(lexEntry["qualifier"]) == str:
            if lexEntry["qualifier"] == "INT":
                self.df_aspect_tokens["intensifier_words"][rowIdx].append(child.text)
                return lexEntry["polarity_strength"]
            elif lexEntry["qualifier"] == "SHI":
                self.df_aspect_tokens["intensifier_words"][rowIdx].append(child.text)
                return -1
            else:
                return 1

        else:
            for i, qualifier in enumerate(lexEntry["qualifier"].values):
                # TODO currently the first qualifier found is taken, without considering which the most fitting one is
                if qualifier == "INT":
                    self.df_aspect_tokens["intensifier_words"][rowIdx].append(
                        child.text)
                    return lexEntry["polarity_strength"][i]
                elif qualifier == "SHI":
                    self.df_aspect_tokens["intensifier_words"][rowIdx].append(
                        child.text)
                    return -1
            return 1

    def calcTotalPolarityStrength(self, child, rowIdx) -> float:
        """
        Calculate the total polarity for a given word

        Args:
            child (spacy.Token): the tokenized word with tagged 'pos_' and 'text'

        Returns:
            polarity_strength (float): the calculated polarity for the given word (child)
        """
        # lemma = self.lemmatizer.find_lemma(child.text, child.pos_)
        polarity_strength = self.checkPolarityAdjective(child, rowIdx)

        # find intensifier in children and multiply their strength to the polarity
        for c in child.children:
            if self.checkValidChild(c, ChildType.INTENSIFIER):
                polarity_strength *= self.checkForIntensifier(c, rowIdx)
        return polarity_strength

    def detectSentiment(self, rowDF: PD.Series) -> None:
        """
        Function to start the other relevent functions

        Args:
            rowDF (PD.Series): row of the Dataframe
        """
        doc = self.nlp(
            " ".join(
                self.df_preprocessed.iloc[rowDF["reviewnumber"]]["tokens"][
                    rowDF["sent_idx"]
                ]
            )
        )

        # print(" ".join([t.text for t in doc]))

        for j, token in enumerate(doc):
            # TODO address words directly by their token index
            if token.text == rowDF["word_found"]:
                for child in token.children:
                    # if child.tag_ == "ADJA":
                    if self.checkValidChild(child, ChildType.DESCRIPTOR):
                        pol_strength = self.calcTotalPolarityStrength(child, rowDF.name)

                        self.df_aspect_tokens["polarity_strength"][rowDF.name].append(
                            pol_strength
                        )

                        self.df_aspect_tokens["sentiment_words"][rowDF.name].append(
                            child.text
                        )
                        return

                for token in doc[j].ancestors:
                    if token.pos_ == "AUX" or token.pos_ == "VERB":
                        for child in token.children:
                            if self.checkValidChild(child, ChildType.DESCRIPTOR):
                                pol_strength = self.calcTotalPolarityStrength(
                                    child, rowDF.name)

                                self.df_aspect_tokens["polarity_strength"][
                                    rowDF.name
                                ].append(pol_strength)

                                self.df_aspect_tokens["sentiment_words"][
                                    rowDF.name
                                ].append(child.text)

                                return

    def convert_polarity(self, qualifier, polarity):
        sentiment_polarity = []
        for i, elem in enumerate(qualifier):
            if elem == "NEG":
                sentiment_polarity.append(polarity[i] * -1)
            else:
                sentiment_polarity.append(polarity[i])
        sentiment_polarity = NP.mean(NP.array(sentiment_polarity))
        return sentiment_polarity

    def createReadableOutput(self, rowDF):
        appenddict = {
            "review_number": rowDF["reviewnumber"],
            "sentiment": self.convert_polarity(
                rowDF["qualifier"], rowDF["polarity_strength"]
            ),
        }

        self.overall_sentiment = self.overall_sentiment.append(
            appenddict, ignore_index=True
        )

    def returnSentimentsforReviews(self) -> PD.DataFrame:
        self.overall_sentiment = PD.DataFrame(columns=["review_text", "sentiment"])
        tqdm.pandas(desc="Calculating Sentiments")
        self.df_aspect_tokens.progress_apply(
            lambda x: self.createReadableOutput(x), axis=1
        )

        self.overall_sentiment = (
            self.overall_sentiment.groupby("review_number").mean().reset_index()
        )
        # print(self.overall_sentiment)
        self.overall_sentiment["review_text"] = self.df_preprocessed["text_normalized"][
            self.overall_sentiment["review_number"].astype(int).tolist()
        ].tolist()

        return self.overall_sentiment

    def run(self) -> bool:
        """
        run all basic functions of the detector

        Returns:
            bool: successful execution of command
        """
        if not self.loadCSVs():
            print("Couldn't load CSV's.")
            return False

        if not self.loadSpacyModel():
            return

        tqdm.pandas(desc="Looking up Sentiments...")
        self.df_aspect_tokens.progress_apply(lambda x: self.detectSentiment(x), axis=1)

    def saveCSV(self, filename: str = "data_aspects_tokens.csv"):
        self.df_aspect_tokens.to_csv(self.path + filename, index=False)


if __name__ == "__main__":
    detector = SentimentDetector()
    detector.run()
    detector.saveCSV()
    # detector.loadCSVs()
    print(detector.df_preprocessed.iloc[30]["text_normalized"])

    # print(detector.returnSentimentsforReviews())
    # detector.overall_sentiment.to_csv("src/data/review_sentiments.csv", index=False)
