# -*- coding: utf-8 -*-
import json
import os
import unittest

import pandas as pd

from src.utils.preprocessing import Preprocessor


class PreprocessingTest(unittest.TestCase):
    def setUp(self):
        self.path = "./src/data/"
        self.preprocessor = Preprocessor(self.path)
        self.test_json = {
            "reviews": [
                {
                    "text": "Das ist für eine Testdatei!",
                    "rating": {
                        "Grafik": 4,
                        "Sound": 5,
                        "Steuerung": 5,
                        "Atmosphäre": 5,
                    },
                }
            ]
        }

    def testValidPath(self):
        self.assertTrue(os.path.isdir(self.preprocessor.path))

    def testNoJsonfiles(self):
        tmpprep = Preprocessor("./")
        self.assertRaises(Exception, lambda: tmpprep.find_jsons())

    def testPrepDefault(self):
        with open(self.path + "testdata.json", "w") as f:
            json.dump(self.test_json, f, indent=4)

        tmpprep = Preprocessor(self.path)
        self.assertTrue([["testdatei"]] == tmpprep.prep().tolist())

    def testPrepNoStopwordRemoval(self):
        with open(self.path + "testdata.json", "w") as f:
            json.dump(self.test_json, f, indent=4)

        tmpprep = Preprocessor(self.path, rmstopwords=False)
        self.assertTrue(
            [["das", "ist", "fuer", "eine", "testdatei"]] == tmpprep.prep().tolist()
        )


if __name__ == "__main__":
    unittest.main()
