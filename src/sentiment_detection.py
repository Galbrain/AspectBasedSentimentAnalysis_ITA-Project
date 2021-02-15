import os
import re

import numpy as NP
import pandas as PD
import requests
from numpy.core.fromnumeric import mean
from numpy.lib.polynomial import polysub
from tqdm import tqdm


class SentimentDetector:
    def __init__(self, path: str = "src/data/", windowSize=5) -> None:
        self.path = path
        self.windowSize = windowSize

        self.df_aspect_tokens = None
        self.df_preprocessed = None
        self.df_lexicon = None

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
                # TODO drop duplicates
                self.df_aspect_tokens.drop_duplicates(inplace=True)
                self.df_aspect_tokens["qualifier"] = PD.NaT
                self.df_aspect_tokens["qualifier"].fillna(
                    {i: [] for i in self.df_aspect_tokens.index}, inplace=True
                )
                self.df_aspect_tokens["polarity_strength"] = PD.NaT
                self.df_aspect_tokens["polarity_strength"].fillna(
                    {i: [] for i in self.df_aspect_tokens.index}, inplace=True
                )

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

                self.df_lexicon = PD.read_csv(
                    self.path + lexiconFilename, index_col="word"
                )

            return True
        except IOError as e:
            print(e)
            return False

    def detectSentiment(self, rowDF: PD.Series) -> None:
        """
        take row of DF and extract review number, uses this review number to create a list of tokens that is [-windowsize:+windowsize].
        then check for every word in that window if it is a key in the sentiment lexicon, if yes save the qualifier in the aspect_token dataset

        Args:
            rowDF (PD.Series): row of Dataframe
        """
        window = self.df_preprocessed.iloc[rowDF["reviewnumber"]]["tokens"][
            rowDF["word_idx"] - self.windowSize : rowDF["word_idx"] + self.windowSize
        ]
        for word in window:
            try:
                if type(self.df_lexicon.loc[word]["qualifier"]) == str:
                    self.df_aspect_tokens["qualifier"][rowDF.name].append(
                        self.df_lexicon.loc[word]["qualifier"]
                    )

                    self.df_aspect_tokens["polarity_strength"][rowDF.name].append(
                        self.df_lexicon.loc[word]["polarity_strength"]
                    )
                else:
                    pass
                    # this should be removed since there should be no dupliate entries in the sentiment lexicon

                    # self.df_aspect_tokens["qualifier"][rowDF.name] = "|".join(
                    #     self.df_lexicon.loc[word]["qualifier"].values
                    # )
                    # self.df_aspect_tokens["polarity_strength"][rowDF.name] = "|".join(
                    #     self.df_lexicon.loc[word]["polarity_strength"].astype(str))
            except KeyError:
                pass

    def createLookupWindow(self) -> None:
        """
        function to vectorize detectSentiment()
        """
        tqdm.pandas(desc="Looking up Sentiments in windows")
        self.df_aspect_tokens.progress_apply(lambda x: self.detectSentiment(x), axis=1)

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
        # self.df_aspect_tokens.apply()

    def saveCSV(self, filename: str = "data_aspects_tokens.csv"):
        self.df_aspect_tokens.to_csv(self.path + filename, index=False)


if __name__ == "__main__":
    detector = SentimentDetector()
    detector.loadCSVs()
    detector.createLookupWindow()
    detector.saveCSV()

    print(detector.returnSentimentsforReviews())
    detector.overall_sentiment.to_csv("src/data/review_sentiments.csv", index=False)
