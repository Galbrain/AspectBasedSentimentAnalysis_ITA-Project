import io
import os
import re
from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile

import nltk
import numpy as NP
import pandas as PD
import requests
import spacy
from germalemma import GermaLemma
from nltk.tokenize import sent_tokenize
from spacy import displacy
from tqdm import tqdm
from utils.senti_ws_wrapper import SentiWSWrapper


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
                ].progress_apply(lambda x: re.sub(r"[\[\]'\s]*", "", x).split(","))

            if self.df_lexicon is None or self.df_lexicon.empty:
                if not os.path.exists(self.path + lexiconFilename):
                    self.downloadLexicon()

                self.df_lexicon = PD.read_csv(self.path + lexiconFilename)
                self.df_lexicon.drop_duplicates(
                    subset=["word", "qualifier"], inplace=True
                )
                self.df_lexicon.set_index("word", inplace=True)

            return True
        except IOError as e:
            print(e)
            return False

    def loadSpacyModel(
        self,
        model: str = "de_core_news_md",
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

    def detectSentiment(self, rowDF: PD.Series, lemmatizer) -> None:
        text = sent_tokenize(
            self.df_preprocessed.iloc[rowDF["reviewnumber"]]["text_normalized"],
            language="german",
        )

        for sentence in text[:100]:
            if rowDF["word_found"] in sentence:
                doc = self.nlp(sentence)

                for token in doc:
                    if token.text == rowDF["word_found"]:
                        for child in token.children:
                            # if child.tag_ == "ADJA":
                            if child.tag_ == "ADJA" and child.pos_ == "ADJ":
                                try:
                                    lemma = self.lemmatizer.find_lemma(
                                        child.text, child.pos_
                                    )
                                    pol_strength = self.df_lexicon.loc[lemma][
                                        "polarity_strength"
                                    ]
                                    if (
                                        type(self.df_lexicon.loc[lemma]["qualifier"])
                                        == str
                                        and self.df_lexicon.loc[lemma]["qualifier"]
                                        == "NEG"
                                    ):
                                        pol_strength *= -1
                                    if (
                                        type(self.df_lexicon.loc[lemma]["qualifier"])
                                        != str
                                        and self.df_lexicon.loc[lemma]["polarity"].any()
                                        == "NEG"
                                    ):
                                        pol_strength *= -1

                                    for c in child.children:
                                        lec = self.df_lexicon.loc[c.text]

                                        if type(lec["qualifier"]) == str:
                                            if lec["qualifier"] == "INT":
                                                pol_strength *= lec["polarity_strength"]
                                                self.df_aspect_tokens[
                                                    "intensifier_words"
                                                ][rowDF.name].append(c.text)
                                            elif (
                                                lec["qualifier"] == "SHI"
                                                and lec["pos"] == "neg"
                                            ):
                                                pol_strength *= -1
                                                self.df_aspect_tokens[
                                                    "intensifier_words"
                                                ][rowDF.name].append(c.text)
                                            else:
                                                pass
                                        else:
                                            for i, elem in enumerate(
                                                lec["qualifier"].values
                                            ):
                                                if elem == "INT":
                                                    pol_strength *= lec[
                                                        "polarity_strength"
                                                    ][i]
                                                    self.df_aspect_tokens[
                                                        "intensifier_words"
                                                    ][rowDF.name].append(c.text)
                                                elif (
                                                    elem == "SHI"
                                                    and lec["pos"][i] == "neg"
                                                ):
                                                    pol_strength *= -1
                                                    self.df_aspect_tokens[
                                                        "intensifier_words"
                                                    ][rowDF.name].append(c.text)

                                    self.df_aspect_tokens["polarity_strength"][
                                        rowDF.name
                                    ].append(pol_strength)

                                    self.df_aspect_tokens["sentiment_words"][
                                        rowDF.name
                                    ].append(child.text)
                                except KeyError:
                                    pass
                            # """
                            # take row of DF and extract review number, uses this review number to create a list of tokens that is [-windowsize:+windowsize].
                            # then check for every word in that window if it is a key in the sentiment lexicon, if yes save the qualifier in the aspect_token dataset

                            # Args:
                            #     rowDF (PD.Series): row of Dataframe
                            # """
                            # window = self.df_preprocessed.iloc[rowDF["reviewnumber"]]["tokens"][
                            #     rowDF["word_idx"] - self.windowSize: rowDF["word_idx"] + self.windowSize
                            # ]

                            # for i, word in enumerate(window):
                            #     try:
                            #         if (
                            #             type(self.df_lexicon.loc[word]["qualifier"]) == str
                            #             and self.df_lexicon.loc[word]["pos"] in ["adj", "neg"]
                            #         ):
                            #             self.df_aspect_tokens["qualifier"][rowDF.name].append(
                            #                 self.df_lexicon.loc[word]["qualifier"]
                            #             )

                            #             self.df_aspect_tokens["polarity_strength"][rowDF.name].append(
                            #                 self.df_lexicon.loc[word]["polarity_strength"]
                            #             )

                            #             self.df_aspect_tokens["sentiment_words"][
                            #                 rowDF.name].append(word)

                            #         else:
                            #             pass
                            #         # elif self.df_lexicon.loc[word]["pos"] in ["adj", "neg"] and self.df_lexicon.loc[window[i]]["qualifier"].any() in ["POS", "NEG"]:
                            #     lexicon_row = self.df_lexicon.loc[word]
                            #     print(lexicon_row.values)
                            #     lexicon_row.drop(
                            #         lexicon_row
                            #         [lexicon_row["qualifier"] in ["POS", "NEG"]],
                            #         inplace=True)
                            #     self.df_aspect_tokens["qualifier"][rowDF.name].append(
                            #     )
                            #     print(lexicon_row)

                            #     self.df_aspect_tokens["polarity_strength"][rowDF.name].append(
                            #         self.df_lexicon.loc[word]["polarity_strength"]
                            #     )

                            #     self.df_aspect_tokens["sentiment_words"][
                            #         rowDF.name].append(word)
                            # this should be removed since there should be no dupliate entries in the sentiment lexicon

                            # self.df_aspect_tokens["qualifier"][rowDF.name] = "|".join(
                            #     self.df_lexicon.loc[word]["qualifier"].values
                            # )
                            # self.df_aspect_tokens["polarity_strength"][rowDF.name] = "|".join(
                            #     self.df_lexicon.loc[word]["polarity_strength"].astype(str))
                            # except KeyError:
                            #     pass

    def createLookupWindow(self) -> None:
        """
        function to vectorize detectSentiment()
        """
        if not self.loadSpacyModel():
            return

        lemmatizer = SentiWSWrapper("src/data/")

        tqdm.pandas(desc="Looking up Sentiments in windows")
        self.df_aspect_tokens.progress_apply(
            lambda x: self.detectSentiment(x, lemmatizer), axis=1
        )

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

    def downloadSW(
        self,
        url="https://pcai056.informatik.uni-leipzig.de/downloads/etc/SentiWS/SentiWS_v2.0.zip",
        path="src/data",
    ):
        with urlopen(url) as zipresp:
            with ZipFile(BytesIO(zipresp.read())) as zfile:
                zfile.extractall(path)

    def run(self) -> bool:
        """
        run all basic functions of the detector

        Returns:
            bool: successful execution of command
        """
        if not self.loadCSVs():
            print("Couldn't load CSV's.")
            return False

        self.downloadSW()

        nltk.download("punkt", download_dir=".venv")

        self.createLookupWindow()

    def saveCSV(self, filename: str = "data_aspects_tokens.csv"):
        self.df_aspect_tokens.to_csv(self.path + filename, index=False)


if __name__ == "__main__":
    detector = SentimentDetector()
    detector.run()
    detector.saveCSV()
    # detector.loadCSVs()
    # print(detector.df_preprocessed.iloc[184]["text_normalized"])

    # print(detector.returnSentimentsforReviews())
    # detector.overall_sentiment.to_csv("src/data/review_sentiments.csv", index=False)
