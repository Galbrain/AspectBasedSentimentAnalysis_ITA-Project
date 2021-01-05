import re

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from requests import get


class WebScraper:
    """
    Webscraper class
    """

    def __init__(self, urls) -> None:
        """
        Init the WebScraper

        Args:
            urls ([str]): List of Urls to be scraped
        """

        self.urls = urls
        self.data = pd.DataFrame()

    def scrape_page(self, url) -> pd.DataFrame:

        return results

    def store_data(self, path: str = "src/data/raw"):
        self.data.to_csv(path + "raw_data.csv")
