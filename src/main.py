import os

import numpy as NP
import requests
from sentiment_detection import SentimentDetector
from utils.aspect_annotator import AspectAnnotator
from utils.preprocessing import Preprocessor
from utils.train import Evaluator
from utils.web_scraper import WebScraper

"""
This script should serve as entrypoint to your program.
Every module or package it relies on has to be imported at the beginning.
The code that is actually executed is the one below 'if __name__ ...' (if run
as script).
"""

do_scraping = True
do_processing = True
do_annotation = True
do_sentimentanalysis = True
do_evaluation = True

Scraper = None
Preper = None
Annotator = None
Detector = None


if __name__ == "__main__":
    if not os.path.exists("src/data/data_raw.csv") or do_scraping:
        urls = NP.loadtxt("src/utils/urls.txt", dtype=str, comments="!")
        Scraper = WebScraper(urls)
        Scraper.startScraping()
        Scraper.storeData()

    if not os.path.exists("src/data/data_preprocessed.csv") or do_processing:
        Preper = Preprocessor(
            lemmatize=False, lower=False, rmnonalphanumeric=False, rmstopwords=False
        )
        Preper.loadSpacyModel(model="de_core_news_md")
        Preper.prep()
        Preper.saveCSV()

    if not os.path.exists("src/data/data_aspects_tokens.csv") or do_annotation:
        Annotator = AspectAnnotator()
        Annotator.loadCSV()
        Annotator.annotate()
        Annotator.saveCSV()

    if do_sentimentanalysis:
        Detector = SentimentDetector()
        Detector.run()
        Detector.saveCSV()

    if do_evaluation:
        evaluator = Evaluator()
        evaluator.generate_train_test()
        evaluator.train_model()
        evaluator.evaluate()
