import json
import re
import time

import pandas as pd
import requests
from bs4 import BeautifulSoup


class WebScraper:
    """
    Webscraper class
    """

    def __init__(self, urls: list[str]):

        self.urls = urls
        self.data = pd.DataFrame

    def scrape_page(self, url: str):
        website = requests.get(url)

        soup = BeautifulSoup(website.text, "html.parser")

        game_id = None

        sticontents = soup.find(
            "div", class_="stiContents stiNoDeco stiOptionsLoadMore mb-3"
        )
        if sticontents:
            game_id = sticontents.get("data-gameid")

            content_url = "https://www.spieletipps.de/gameopinion/opinion-xhr/"
            game_data = {"id": game_id, "offset": 0, "limit": 10}

            content = requests.get(content_url, data=game_data)

            print(content.text)

        return None

    def start_scraping(self):
        for url in self.urls:
            df_review = self.scrape_page(url)
            df_review = {}
            self.data.append(df_review, ignore_index=True)

    def store_data(self, path: str = "src/data/"):
        self.data.to_csv(path + "raw_data.csv", index=False)

    # web_scraper.store_data() # currently not neccessary, since start_scraping already stores gathered data for each url
