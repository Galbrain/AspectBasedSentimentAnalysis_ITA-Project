import os
import unittest

from src.utils.preprocessing import Preprocessor


class PreprocessingTest(unittest.TestCase):

    def setUp(self):
        self.preprocessor = Preprocessor("./src/data/raw/")

    def testValidPath(self):
        self.assertTrue(os.path.isdir(self.preprocessor.path))

    def testNoJsonfiles(self):
        tmpprep = Preprocessor("./")
        self.assertRaises(Exception, lambda: tmpprep.import_jsons())


if __name__ == "__main__":
    unittest.main()
