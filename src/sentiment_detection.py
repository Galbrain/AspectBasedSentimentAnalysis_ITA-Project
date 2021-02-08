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

            if self.df_preprocessed is None or self.df_preprocessed.empty:
                self.df_preprocessed = PD.read_csv(self.path + preprocessedFilename)
                print("Applying Datatype Transformations....")
                self.df_preprocessed["tokens"] = (
                    self.df_preprocessed["tokens"]
                    .str.split(",", expand=True)
                    .replace(r"[\[\]]", "", regex=True)
                    .astype(str)
                    .values.tolist()
                )
                print("Done")

            if self.df_lexicon is None or self.df_lexicon.empty:
                if not os.path.exists(self.path + lexiconFilename):
                    self.downloadLexicon()

                self.df_lexicon = PD.read_csv(self.path + lexiconFilename)

            return True
        except IOError as e:
            print(e)
            return False

    def detectSentiment(self) -> None:
        pass

    def createWindow(self) -> None:
        pass

    def run(self) -> bool:
        if not self.loadCSVs():
            print("Couldn't load CSV's.")
            return False
        pass
        # self.df_aspect_tokens.apply()


if __name__ == "__main__":
    detector = SentimentDetector()
    detector.loadCSVs()
    print(detector.df_preprocessed)
