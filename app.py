import streamlit as st
import numpy as np
from algorithm import NeuralNetwork
from utils import get_xor_data
import matplotlib.pyplot as plt

st.set_page_config(page_title="Neural Network Visualization", layout="centered")

st.title("Neural Network Visualization - XOR Problem")

# Sidebar for inputs
st.sidebar.header("Training Parameters")
epochs = st.sidebar.slider("Epochs", min_value=100, max_value=5000, value=1000, step=100)
learning_rate = st.sidebar.slider("Learning Rate", min_value=0.01, max_value=1.0, value=0.1, step=0.01)
hidden_neurons = st.sidebar.slider("Hidden Layer Neurons", min_value=1, max_value=10, value=2)

# Load XOR dataset
X, y = get_xor_data()

# Display input data
st.write("### XOR Dataset Inputs")
st.write(X)
st.write("### Labels")
st.write(y)

if st.button("Train Network"):
    nn = NeuralNetwork(input_size=2, hidden_size=hidden_neurons, output_size=1, learning_rate=learning_rate)
    losses = nn.train(X, y, epochs=epochs)
    
    # Plot loss curve
    fig, ax = plt.subplots()
    ax.plot(losses)
    ax.set_title("Training Loss Over Epochs")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    st.pyplot(fig)
    
    # Display predictions
    predictions = nn.predict(X)
    st.write("### Predictions on XOR Inputs")
    st.write(predictions)
    
    # Display final weights and biases
    st.write("### Final Weights and Biases")
    st.write("Weights 1 (Input to Hidden):")
    st.write(nn.W1)
    st.write("Biases 1:")
    st.write(nn.b1)
    st.write("Weights 2 (Hidden to Output):")
    st.write(nn.W2)
    st.write("Biases 2:")
    st.write(nn.b2)
