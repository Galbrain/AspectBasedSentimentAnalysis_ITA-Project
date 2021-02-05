# import re

# import numpy as np
# import pandas as pd
# import requests
# from bs4 import BeautifulSoup
# from requests import get


# class WebScraper:
# """
# Webscraper class
# """

# def __init__(self, urls) -> None:
# """
# Init the WebScraper

# Args:
# urls ([str]): List of Urls to be scraped
# """

# self.urls = urls
# self.data = pd.DataFrame()

# def scrape_page(self, url) -> pd.DataFrame:

# results = requests.get(url)

# soup = BeautifulSoup(results.text, "html.parser")

# author = []
# review_text = []
# review_rating_grafik = []
# review_rating_sound = []
# review_rating_steuerung = []
# review_rating_atmosphaere = []

# review = soup.find_all('div', class_="stiOpinionBox")

# return results

# def store_data(self, path: str = "src/data/raw"):
# self.data.to_csv(path + "raw_data.csv")


# if __name__ == "__main__":
# scraper = WebScraper(["https://www.spieletipps.de/game/dead-space/#meinungen"])
