# -*- coding: utf-8 -*-
import os
import unittest

from src.utils.aspect_annotator import AspectAnnotator


class AnnotatorTest(unittest.TestCase):
    def setUp(self):
        self.path = "./tests/data/"
        self.annotator = AspectAnnotator(self.path)
        self.annotator.loadCSV("test_preprocessed.csv")

    def testValidPath(self):
        self.assertTrue((os.path.isdir(self.annotator.path)))

    def testLoadCSV(self):
        self.assertTrue([["schöne", "testdatei"]] == self.annotator.data["tokens"][0])

    def testAnnotate(self):
        self.annotator.annotate()
        self.assertTrue(self.annotator.df.iloc[0]["aspect"] == "Grafik")


if __name__ == "__main__":
    unittest.main()
