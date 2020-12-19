import unittest

from src.utils.preprocessing import Preprocessor


class ReproTest(unittest.TestCase):

    def setup(self):
        self.preprocessor = Preprocessor()

    def no_path(self):
        self.assertTrue(not self.preprocessor.path)


if __name__ == "__main__":
    unittest.main()
