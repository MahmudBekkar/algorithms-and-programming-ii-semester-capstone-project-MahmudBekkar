import streamlit as st
import numpy as np
from algorithm import SimpleNeuralNetwork
import matplotlib.pyplot as plt

st.title("Neural Network Visualization")

input_size = st.slider("Input Neurons", 2, 10, 2)
hidden_size = st.slider("Hidden Neurons", 2, 10, 4)
output_size = st.slider("Output Neurons", 1, 3, 1)
epochs = st.slider("Training Epochs", 10, 500, 100)
lr = st.slider("Learning Rate", 0.01, 1.0, 0.1)

# Sample dataset (XOR)
X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[1],[1],[0]])

nn = SimpleNeuralNetwork(input_size, hidden_size, output_size, learning_rate=lr)

losses = []
for epoch in range(epochs):
    output = nn.forward(X)
    loss = np.mean((y - output)**2)
    losses.append(loss)
    nn.backward(X, y)

st.line_chart(losses)

st.write("Final Output:")
st.write(output)
