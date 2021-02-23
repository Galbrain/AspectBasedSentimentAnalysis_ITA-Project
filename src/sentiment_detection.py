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
            bool: sucessful execution
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
                print(self.df_preprocessed)

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
            if (child.tag_ == "ADJA" and child.pos_ == "ADJ") or (child.pos_ == "ADV"):
                return True
            return False
        elif childType == ChildType.INTENSIFIER:
            if child.pos_ == "ADJ" or child.pos_ == "ADV":
                return True
            return False
        else:
            print("Wrong childType.")
            return False

    def checkPolarityAdjective(self, child) -> float:
        """
        check if the given word has an entry in the sentiment lexicon and return given polarity strength

        Args:
            child (spacy.Token): tokenized word with tagged 'pos_' and 'text'

        Returns:
            pol_strength (float): polarity_strength of given word found in sentiment lexicon
        """
        lemma = self.lemmatizer.find_lemma(
            child.text.replace(r"[^\w]*", ""), child.pos_
        )

        # catch words that are not in the sentiment lexicon
        try:
            lexEntry = self.df_lexicon.loc[lemma]
        except KeyError:
            # print('Word not found in Sentiment Lexicon:', lemma)
            return 0

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
            return 0

    def checkForIntensifier(self, child) -> float:
        """
        For a given spacy.Token (child) check if any of the children is an intensifier and if so, return their polarity_strength

        Args:
            child (spacy.Token): tokenized word with tagged 'pos_' and 'text'

        Returns:
            polarity_multiplier (float): polarity_multiplier of found intensifier word
        """
        lemma = self.lemmatizer.find_lemma(child.text, child.pos_)

        try:
            lexEntry = self.df_lexicon.loc[lemma]
            # print('Valid lemma in intensifier', lemma)
        except KeyError:
            # print("Intensifier not in lexicon", lemma)
            return 1

        if type(lexEntry["qualifier"]) == str:
            if lexEntry["qualifier"] == "INT":
                return lexEntry["polarity_strength"]
            elif lexEntry["qualifier"] == "SHI":
                return -1
            else:
                return 1

        else:
            for i, qualifier in enumerate(lexEntry["qualifier"].values):
                # TODO currently the first qualifier found is taken, without considering which the most fitting one is
                if qualifier == "INT":
                    return lexEntry["polarity_strength"][i]
                elif qualifier == "SHI":
                    return -1
            return 1

    def calcTotalPolarityStrength(self, child) -> float:
        """
        Calculate the total polarity for a given word

        Args:
            child (spacy.Token): the tokenized word with tagged 'pos_' and 'text'

        Returns:
            polarity_strength (float): the calculated polarity for the given word (child)
        """
        # lemma = self.lemmatizer.find_lemma(child.text, child.pos_)
        polarity_strength = self.checkPolarityAdjective(child)

        # find intensifier in children and multiply their strength to the polarity
        for c in child.children:
            if self.checkValidChild(c, ChildType.INTENSIFIER):
                polarity_strength *= self.checkForIntensifier(c)
        return polarity_strength

    def detectSentiment(self, rowDF: PD.Series) -> None:
        doc = self.nlp(
            " ".join(
                self.df_preprocessed.iloc[rowDF["reviewnumber"]]["tokens"][
                    rowDF["sent_idx"]
                ]
            )
        )

        for j, token in enumerate(doc):
            if token.text == rowDF["word_found"]:
                for child in token.children:
                    # if child.tag_ == "ADJA":
                    if self.checkValidChild(child, ChildType.DESCRIPTOR):
                        pol_strength = self.calcTotalPolarityStrength(child)

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
                                pol_strength = self.calcTotalPolarityStrength(child)

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

        true_labels = list()
        for index, row in self.df_aspect_tokens.iterrows():
            true_labels.append(
                self.df_preprocessed.iloc[row['reviewnumber']]
                [self.df_aspect_tokens.iloc[index]['aspect']])
        self.df_aspect_tokens['true_label'] = true_labels

        tqdm.pandas(desc="Looking up Sentiments...")
        self.df_aspect_tokens.progress_apply(lambda x: self.detectSentiment(x), axis=1)

    def saveCSV(self, filename: str = "data_aspects_tokens.csv"):
        self.df_aspect_tokens.to_csv(self.path + filename, index=False)


if __name__ == "__main__":
    detector = SentimentDetector()
    detector.run()
    detector.saveCSV()
    # detector.loadCSVs()
    # print(detector.df_preprocessed.iloc[14]["text_normalized"])

    # print(detector.returnSentimentsforReviews())
    # detector.overall_sentiment.to_csv("src/data/review_sentiments.csv", index=False)
