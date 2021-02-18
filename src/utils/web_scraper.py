import json

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

    def scrape_page(self, url: str) -> None:
        """
        Extract the Game_id from the Website link, then use the Spieletipps API to get the content

        Args:
            url (str): url of Game
        """
        website = requests.get(url)

        soup = BeautifulSoup(website.text, "html.parser")

        game_id = None

        # find the "more" button
        sticontents = soup.find(
            "div", class_="stiContents stiNoDeco stiOptionsLoadMore mb-3"
        )

        # if there is a more button extract the game_id
        if sticontents:
            game_id = sticontents.get("data-gameid")

            # get the content once to find out how many total reviews there are
            content_url = "https://www.spieletipps.de/gameopinion/opinion-xhr/"
            offset = 0
            game_data = {"id": game_id, "offset": offset, "limit": 1}

            content = requests.get(content_url, params=game_data)
            content_json = json.loads(content.text)

            total_num_reviews = content_json["response"]["data"]["more"]

            # iterate over the total reviews in steps of 10
            while total_num_reviews - offset > 0:
                game_data = {"id": game_id, "offset": offset, "limit": 10}
                offset += 10

                content = requests.get(content_url, params=game_data)
                content_json = json.loads(content.text)

                # create a dict for every review and append the data to the dataframe
                for i in content_json["response"]["data"]["data"]:
                    reviewDict = {
                        "titel": i["game"],
                        "review_text_raw": i["content"]
                        .replace("<br />", "")
                        .replace("&quot;", "")
                        .replace("\n", ""),
                        "Grafik": i["ratingDetail"]["score_graphics"],
                        "Sound": i["ratingDetail"]["score_sound"],
                        "Steuerung": i["ratingDetail"]["score_gameplay"],
                        "Atmosphäre": i["ratingDetail"]["score_atmosphere"],
                    }
                    self.data = self.data.append(reviewDict, ignore_index=True)

    def start_scraping(self):
        for url in tqdm(self.urls, desc="Scraping Urls.."):
            self.scrape_page(url)

    def store_data(self, path: str = "src/data/"):
        self.data.to_csv(path + "data_raw.csv", index=False)

    # web_scraper.store_data() # currently not neccessary, since start_scraping already stores gathered data for each url
