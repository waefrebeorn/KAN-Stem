import unittest
import torch
from src.kan.KAN import KAN

class TestKAN(unittest.TestCase):

    def setUp(self):
        self.model = KAN(width=[2, 5, 1], grid=5, k=3)

    def test_kan_initialization(self):
        self.assertIsInstance(self.model, KAN)

    def test_kan_forward(self):
        x = torch.rand(10, 2)
        y = self.model.forward(x)
        self.assertEqual(y.shape, (10, 1))

if __name__ == '__main__':
    unittest.main()
