import unittest
from modules import main_script


class TestMainScript(unittest.TestCase):
    def test_sample(self):
        self.assertEqual(main_script.sample_function(), "expected_output")


if __name__ == "__main__":
    unittest.main()
