import os
import re

import numpy as NP
import pandas as PD
import requests
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
        try:
            if self.df_lexicon is None or self.df_aspect_tokens.empty:
                self.df_aspect_tokens = PD.read_csv(self.path + tokenFilename)
                self.df_aspect_tokens["qualifier"] = NP.nan

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
        window = self.df_preprocessed.iloc[rowDF["reviewnumber"]]["tokens"][
            rowDF["word_idx"] - self.windowSize : rowDF["word_idx"] + self.windowSize
        ]
        for word in window:
            try:
                # print(self.df_aspect_tokens.iloc[rowDF.name])
                if type(self.df_lexicon.loc[word]["qualifier"]) == str:
                    self.df_aspect_tokens["qualifier"][
                        rowDF.name
                    ] = self.df_lexicon.loc[word]["qualifier"]
                else:
                    self.df_aspect_tokens["qualifier"][rowDF.name] = "|".join(
                        self.df_lexicon.loc[word]["qualifier"].values
                    )
            except KeyError:
                pass

    def createLookupWindow(self) -> None:
        tqdm.pandas(desc="Looking up Sentiments in windows")
        self.df_aspect_tokens.progress_apply(lambda x: self.detectSentiment(x), axis=1)

    def run(self) -> bool:
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
    print(detector.df_aspect_tokens)
