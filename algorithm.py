import numpy as np

class SimpleNeuralNetwork:
    """
    A simple neural network with one hidden layer.
    """

    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        self.W1 = np.random.randn(self.input_size, self.hidden_size)
        self.b1 = np.zeros((1, self.hidden_size))

        self.W2 = np.random.randn(self.hidden_size, self.output_size)
        self.b2 = np.zeros((1, self.output_size))

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_derivative(self, z):
        return z * (1 - z)

    def forward_propagation(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2

    def back_propagation(self, X, y):
        m = X.shape[0]
        error = self.a2 - y

        dW2 = np.dot(self.a1.T, error * self.sigmoid_derivative(self.a2)) / m
        db2 = np.sum(error * self.sigmoid_derivative(self.a2), axis=0, keepdims=True) / m

        delta_hidden = np.dot(error * self.sigmoid_derivative(self.a2), self.W2.T) * self.sigmoid_derivative(self.a1)
        dW1 = np.dot(X.T, delta_hidden) / m
        db1 = np.sum(delta_hidden, axis=0, keepdims=True) / m

        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2

    def train(self, X, y, epochs=100):
        for epoch in range(epochs):
            self.forward_propagation(X)
            self.back_propagation(X, y)

    def predict(self, X):
        return self.forward_propagation(X)
