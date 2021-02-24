# -*- coding: utf-8 -*-
import json
import os
import unittest

import pandas as pd

from src.utils.preprocessing import Preprocessor


class PreprocessingTest(unittest.TestCase):
    def setUp(self):
        self.path = "./tests/data/"
        self.preprocessor = Preprocessor(self.path)

    def testValidPath(self):
        self.assertTrue(os.path.isdir(self.preprocessor.path))

    def testPrepDefault(self):
        tmpprep = Preprocessor(self.path)
        tmpprep.loadCSV("test.csv")
        self.assertTrue(tmpprep.prep())

    def testPrepNoStopwordRemoval(self):
        tmpprep = Preprocessor(self.path, rmstopwords=False)
        tmpprep.loadCSV("test.csv")
        tmpprep.prep()
        self.assertTrue(
            [["das", "ist", "für", "eine", "schöne", "testdatei"]]
            == tmpprep.data["tokens"].tolist()[0]
        )

if __name__ == "__main__":
    unittest.main()
