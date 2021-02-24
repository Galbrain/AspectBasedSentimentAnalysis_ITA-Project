import json
from os import replace

import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm


class WebScraper:
    """
    Webscraper class
    """

    def __init__(self, urls: list[str]):
        self.urls = urls
        self.data = pd.DataFrame(
            columns=[
                "titel",
                "review_text_raw",
                "Grafik",
                "Sound",
                "Steuerung",
                "Atmosphäre",
            ]
        )

    def parseResponse(self, content: dict) -> None:
        # create a dict for every review and append the data to the dataframe
        for i in content["response"]["data"]["data"]:
            reviewDict = {
                "titel": i["game"],
                "review_text_raw": i["content"]
                .replace("<br />", "")
                .replace("&quot;", "")
                .replace("\n", " ")
                .replace("\r", " "),
                "Grafik": i["ratingDetail"]["score_graphics"],
                "Sound": i["ratingDetail"]["score_sound"],
                "Steuerung": i["ratingDetail"]["score_gameplay"],
                "Atmosphäre": i["ratingDetail"]["score_atmosphere"],
            }
            self.data = self.data.append(reviewDict, ignore_index=True)

    def getResponse(self, game_id: int, offset: int, limit: int) -> dict:
        content_url = "https://www.spieletipps.de/gameopinion/opinion-xhr/"
        game_data = {"id": game_id, "offset": offset, "limit": limit}

        content = requests.get(content_url, params=game_data)
        return json.loads(content.text)

    def getGameID(self, url: str) -> int:
        website = requests.get(url)
        soup = BeautifulSoup(website.text, "html.parser")

        # find the "more" button
        sticontents = soup.find(
            "div", class_="stiContents stiNoDeco stiOptionsLoadMore mb-3"
        )
        if sticontents:
            return sticontents.get("data-gameid")
        else:
            return None

    def scrapePage(self, url: str) -> None:
        """
        Extract the Game_id from the Website link, then use the Spieletipps API to get the content

        Args:
            url (str): url of Game
        """

        # if there is a more button extract the game_id
        game_id = self.getGameID(url)
        if not game_id:
            return

        # get the content once to find out how many total reviews there are
        content = self.getResponse(game_id, 0, 1)
        total_num_reviews = content["response"]["data"]["more"]

        # iterate over the total reviews in steps of 10
        offset = 0
        while total_num_reviews - offset > 0:
            offset += 10
            content = self.getResponse(game_id, offset, 10)
            self.parseResponse(content)

    def startScraping(self):
        for url in tqdm(self.urls, desc="Scraping Urls.."):
            self.scrapePage(url)

    def storeData(self, path: str = "src/data/", filename: str = "data_raw.csv"):
        self.data.to_csv(path + filename, index=False)

    # web_scraper.store_data() # currently not neccessary, since start_scraping already stores gathered data for each url
