import unittest
import numpy as np
from algorithm import SimpleNeuralNetwork

class TestNN(unittest.TestCase):
    def test_forward_shape(self):
        nn = SimpleNeuralNetwork(2, 4, 1)
        X = np.array([[0, 1]])
        output = nn.forward(X)
        self.assertEqual(output.shape, (1, 1))

    def test_output_range(self):
        nn = SimpleNeuralNetwork(2, 4, 1)
        X = np.array([[1, 1]])
        output = nn.forward(X)
        self.assertTrue((output >= 0).all() and (output <= 1).all())

if __name__ == "__main__":
    unittest.main()
