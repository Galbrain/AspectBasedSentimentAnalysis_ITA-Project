import os

import numpy as NP
import requests
from sentiment_detection import SentimentDetector
from utils.aspect_annotator import AspectAnnotator
from utils.preprocessing import Preprocessor
from utils.web_scraper import WebScraper

"""
This script should serve as entrypoint to your program.
Every module or package it relies on has to be imported at the beginning.
The code that is actually executed is the one below 'if __name__ ...' (if run
as script).
"""
do_scraping = False
do_processing = True
do_annotation = True
do_sentimentanalysis = False

Scraper = None
Preper = None
Anotator = None
Detector = None

if __name__ == "__main__":
    if not os.path.exists("src/data/data_raw.csv") or do_scraping:
        urls = NP.loadtxt("src/utils/urls.txt", dtype=str, comments="!")
        Scraper = WebScraper(urls)
        Scraper.start_scraping()
        Scraper.store_data()

    if not os.path.exists("src/data/data_preprocessed.csv") or do_processing:
        Preper = Preprocessor(
            lemmatize=False, lower=False, rmnonalphanumeric=False, rmstopwords=False
        )
        Preper.loadSpacyModel(model="de_core_news_md")
        Preper.prep()
        Preper.saveCSV()

    if not os.path.exists("src/data/data_aspects_tokens.csv") or do_annotation:
        Anotator = AspectAnnotator()
        Anotator.loadCSV()
        Anotator.annotate()
        Anotator.saveCSV()

    if do_sentimentanalysis:
        Detector = SentimentDetector()
        if Preper:
            Detector.df_preprocessed = Preper.data
        Detector.run()
        Detector.saveCSV()
