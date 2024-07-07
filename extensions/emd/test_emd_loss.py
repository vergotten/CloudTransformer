import unittest
import torch
import numpy as np
from emd import earth_mover_distance


class TestEarthMoverDistance(unittest.TestCase):
    def setUp(self):
        self.x = torch.rand(4, 64, 3).cuda()
        self.y = torch.rand(4, 128, 3).cuda()

    def test_earth_mover_distance(self):
        emd = earth_mover_distance(transpose=False)  # Create an instance of the class
        result = emd.forward(self.x, self.y)  # Call the forward method on the instance
        self.assertIsInstance(result, torch.Tensor)

    def test_earth_mover_distance_transpose(self):
        emd = earth_mover_distance(transpose=True)  # Create an instance of the class with transpose=True
        result = emd.forward(self.x, self.y)  # Call the forward method on the instance
        self.assertIsInstance(result, torch.Tensor)

    def test_earth_mover_distance_zero_input(self):
        zero_tensor = torch.zeros(4, 64, 3).cuda()
        emd = earth_mover_distance(transpose=False)
        result = emd.forward(zero_tensor, self.y)
        self.assertIsInstance(result, torch.Tensor)


if __name__ == '__main__':
    unittest.main()  # This will run all the test methods
