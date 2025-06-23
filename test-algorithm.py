import unittest
import numpy as np
from algorithm import NeuralNetwork
from utils import get_xor_data

class TestNeuralNetwork(unittest.TestCase):
    def test_xor_training(self):
        X, y = get_xor_data()
        nn = NeuralNetwork(input_size=2, hidden_size=2, output_size=1, learning_rate=0.1)
        losses = nn.train(X, y, epochs=5000)
        predictions = nn.predict(X)
        expected = y
        np.testing.assert_array_equal(predictions, expected)

if __name__ == "__main__":
    unittest.main()
