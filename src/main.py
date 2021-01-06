import os

import numpy as NP
from utils.web_scraper import WebScraper

"""
This script should serve as entrypoint to your program.
Every module or package it relies on has to be imported at the beginning.
The code that is actually executed is the one below 'if __name__ ...' (if run
as script).
"""

if __name__ == "__main__":
    if not os.path.exists("src/data/data_raw.csv"):
        urls = NP.loadtxt("src/utils/urls.txt", dtype=str, comments="!")
        # Scraper = WebScraper(list(urls[0]))
        # Scraper.start_scraping()
        # Scraper.store_data()
