import unittest
import numpy as np
from algorithm import SimpleNeuralNetwork

class TestNN(unittest.TestCase):
    def test_shape(self):
        X = np.random.rand(10, 2)
        y = np.random.randint(0, 2, size=(10, 1))
        model = SimpleNeuralNetwork(2, 5, 1)
        output = model.predict(X)
        self.assertEqual(output.shape, (10, 1))

if __name__ == '__main__':
    unittest.main()
