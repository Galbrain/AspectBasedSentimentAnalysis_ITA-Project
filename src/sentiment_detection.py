import os

import numpy as NP
import pandas as PD
import requests
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
        url="https://raw.githubusercontent.com/sebastiansauer/pradadata/master/data-raw/germanlex.csv",
        chunk_size=128,
    ) -> None:
        """
        Download sentiment lexicon.

        Args:
            filename (str, optional):  Defaults to "sentimentLexicon.csv".
            url (str, optional):  Defaults to "https://raw.githubusercontent.com/sebastiansauer/pradadata/master/data-raw/germanlex.csv".
            chunk_size (int, optional): Defines chunk size for downloads of bigger files. Defaults to 128.
        """
        # TODO progress bar
        r = requests.get(url, stream=True)

        file_size = int(r.headers.get("Content-Length", None))
        num_bars = NP.ceil(file_size / (chunk_size))

        downloadProgress = tqdm(total=num_bars, desc="Downloading Lexicon...", unit="B")
        with open(self.path + filename, "wb") as fd:
            for chunk in r.iter_content(chunk_size=chunk_size):
                downloadProgress.update(chunk)
                fd.write(chunk)
        downloadProgress.close()

    def loadCSVs(
        self,
        tokenFilename="data_aspects_tokens.csv",
        preprocessedFilename="data_preprocessed.csv",
        lexiconFilename="sentiment_lexicon.csv",
    ) -> bool:
        try:
            self.df_aspect_tokens = PD.read_csv(self.path + tokenFilename)
            self.df_aspect_tokens["tokens"] = (
                self.df_aspect_tokens["tokens"]
                .str.split(",", expand=True)
                .replace(r"[\[\]]", "", regex=True)
                .astype(str)
                .values.tolist()
            )
            self.df_preprocessed = PD.read_csv(self.path + preprocessedFilename)
            if not os.path.exists(self.path + lexiconFilename):
                self.downloadLexicon()

            self.df_lexicon = PD.read_csv(self.path + lexiconFilename)
            return True
        except IOError as e:
            print(e)
            return False

    def detectSentiment(self) -> None:
        pass

    def createWindow(self):
        pass

    def run(self):
        if not self.loadCSVs():
            print("Couldn't load CSV's.")
            return False

        self.df_aspect_tokens.apply()


if __name__ == "__main__":
    detector = SentimentDetector()
    detector.loadCSVs()
    print(detector.df_preprocessed)
