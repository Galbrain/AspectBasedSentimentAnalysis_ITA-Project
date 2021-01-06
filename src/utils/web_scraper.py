import json
import re
import time

import pandas as pd
import requests
import selenium
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.common.exceptions import (ElementClickInterceptedException,
                                        ElementNotInteractableException,
                                        NoSuchElementException)
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait


class WebScraper:
    """
    Webscraper class
    """

    def __init__(self, urls: [str]):

        self.urls = urls
        self.driver = webdriver.Firefox(executable_path=r".venv/geckodriver.exe")
        self.driver.implicitly_wait(10)
        self.data = pd.DataFrame

    def scrape_page(self, url: str):

        driver = self.driver
        driver.get(url)

        soup = BeautifulSoup(driver.page_source, "html.parser")

        iframes_size = len(driver.find_elements_by_xpath("//iframe"))

        for iframe_index in range(iframes_size):

            driver.switch_to.frame(iframe_index)
            button_path = "/html/body/div/div[3]/div[2]/button"
            try:
                button = driver.find_element_by_xpath(button_path)
                button.click()
                driver.switch_to.default_content()
            except ElementNotInteractableException:  # TODO: THIS IS PROBABLY WRONG
                driver.switch_to.default_content()
                continue
            break

        title = soup.find("h1").get_text()
        while True:
            try:
                button = driver.find_element_by_partial_link_text("weitere Artikel")

            except NoSuchElementException:
                try:
                    button_path = "//span[@class='chevron-down']"
                    button = driver.find_element_by_xpath(button_path)
                except NoSuchElementException:
                    print("except")
                    break

            soup = BeautifulSoup(driver.page_source, "html.parser")
            try:
                button.click()
            except ElementClickInterceptedException:
                self.driver.implicitly_wait(10)
                button_path = "//button[@class='cleverpush-confirm-btn cleverpush-confirm-btn-deny'][.='später']"
                button = driver.find_element_by_xpath(button_path)
                button.click()

        opinions_tags = soup.find_all("div", class_="collapse")

        review_summaries = pd.DataFrame(columns=['title', 'review_text_raw', 'rating'])

        for opinion_tag in opinions_tags:

            stars = dict()
            for i, aspect in enumerate(opinion_tag.find_all("dt")):
                stars_container = opinion_tag.find_all("dd")
                stars_amount1 = len(stars_container[i].find_all("i", class_="fa-star"))
                stars_amount2 = len(
                    stars_container[i].find_all("img", class_="stiDetailratingStarOn")
                )
                stars_amount = max(stars_amount1, stars_amount2)
                stars[aspect.get_text().strip(":")] = stars_amount

            dls = opinion_tag.find_all("dl")
            for dl in dls:
                dl.decompose()

            review_summary = {
                'title': title,
                "review_text_raw": opinion_tag.get_text(),
                "rating": stars}
            review_summaries.append(review_summary, ignore_index=True)

        return review_summaries

    def start_scraping(self):
        for url in self.urls:
            df_review = self.scrape_page(url)
            self.data.append(df_review, ignore_index=True)

    def store_data(self, path: str = "src/data/raw/"):
        self.data.to_csv(path + "raw_data.csv", index=False)


if __name__ == "__main__":
    urls = [
        "https://www.spieletipps.de/game/dead-space/#meinungen",
        #  "https://www.spieletipps.de/game/dead-space-2/#meinungen",
        # "https://www.spieletipps.de/game/gta-san-andreas/#meinungen",
        # "https://www.spieletipps.de/game/dishonored/",
        # "https://www.spieletipps.de/game/uncharted-2/#meinungen",
        # "https://www.spieletipps.de/game/sims-3/#meinungen",
        # "https://www.spieletipps.de/game/arkham-city/#meinungen",
        # "https://www.spieletipps.de/game/borderlands/#meinungen",
        # "https://www.spieletipps.de/game/diablo-3/#meinungen",
        # "https://www.spieletipps.de/game/ac-4/#meinungen",
        # "https://www.spieletipps.de/game/far-cry-3/#meinungen",
        # "https://www.spieletipps.de/game/skyrim/#meinungen",
        # "https://www.spieletipps.de/game/gta-5/#meinungen",
        # "https://www.spieletipps.de/game/demons-souls/#meinungen",
        # "https://www.spieletipps.de/game/tomb-raider/#meinungen",
        # "https://www.spieletipps.de/game/last-of-us-remastered/#meinungen",
        # "https://www.spieletipps.de/game/new-super-mario-bros-1/#meinungen",
        # "https://www.spieletipps.de/game/super-mario-galaxy-2/#meinungen",
        # "https://www.spieletipps.de/game/super-mario-galaxy/#meinungen",
        # "https://www.spieletipps.de/game/mario-kart/#meinungen",
        # "https://www.spieletipps.de/game/gta-vice-city/#meinungen",
        # "https://www.spieletipps.de/game/grand-theft-auto-3/#meinungen",
        # "https://www.spieletipps.de/game/rocket-league/#meinungen",
        # "https://www.spieletipps.de/game/anno-1404/#meinungen",
        # "https://www.spieletipps.de/game/ff-x-1/#meinungen",
        # "https://www.spieletipps.de/game/witcher-3/#meinungen",
        # "https://www.spieletipps.de/game/uncharted/#meinungen",
        # " https://www.spieletipps.de/game/nfs-hot-pursuit/#meinungen",
        # "https://www.spieletipps.de/game/portal-2/#meinungen",
        # "https://www.spieletipps.de/game/zelda-skyward-sword/#meinungen",
        # "https://www.spieletipps.de/game/zelda-twilight-princess/#meinungen",
        # "https://www.spieletipps.de/game/super-smash-bros-brawl/#meinungen",
        # "https://www.spieletipps.de/game/mass-effect-2/#meinungen",
        # "https://www.spieletipps.de/game/ac-wild-world/#meinungen",
        # "https://www.spieletipps.de/game/monster-hunter-3/#meinungen",
        # "https://www.spieletipps.de/game/ac-2/#meinungen",
        # "https://www.spieletipps.de/game/assassins-creed/#meinungen",
        # "https://www.spieletipps.de/game/assassins-creed-3/#meinungen",
        # "https://www.spieletipps.de/game/beyond-two-souls/#meinungen",
        # "https://www.spieletipps.de/game/heavy-rain/#meinungen",
        # "https://www.spieletipps.de/game/lbp-2/#meinungen",
        # "https://www.spieletipps.de/game/lbp/#meinungen",
        # "https://www.spieletipps.de/game/dmc-devil-may-cry/#meinungen",
        # "https://www.spieletipps.de/game/halo-4/#meinungen",
        # "https://www.spieletipps.de/game/gothic/#meinungen",
        # "https://www.spieletipps.de/game/minecraft/#meinungen",
    ]

    web_scraper = WebScraper(urls)

    web_scraper.start_scraping()

    web_scraper.store_data()

    # web_scraper.store_data() # currently not neccessary, since start_scraping already stores gathered data for each url
