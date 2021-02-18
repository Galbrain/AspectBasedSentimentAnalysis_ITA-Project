import os
import unittest

from src.utils.web_scraper import WebScraper


class WebScraperTest(unittest.TestCase):
    def setUp(self) -> None:
        self.WebScraper = WebScraper(["https://www.spieletipps.de/game/dishonored/"])
        self.gameid = self.WebScraper.getGameID(self.WebScraper.urls[0])

    def testGetGameID(self):
        self.assertTrue(self.gameid == "107281")

    def testGetResponse(self):
        response = self.WebScraper.getResponse(self.gameid, 0, 1)
        self.assertEqual(type(response), dict)

    def testParseResponse(self):
        response = self.WebScraper.getResponse(self.gameid, 0, 10)
        self.WebScraper.parseResponse(response)
        self.assertTrue(len(self.WebScraper.data) == 10)

    def testStoreData(self):
        response = self.WebScraper.getResponse(self.gameid, 0, 10)
        self.WebScraper.parseResponse(response)
        self.WebScraper.storeData("tests/data/", "test_raw.csv")
        self.assertTrue(os.path.exists("tests/data/test_raw.csv"))
