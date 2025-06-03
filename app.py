import streamlit as st
import numpy as np
from algorithm import NeuralNetwork

st.title("Simple Neural Network Visualization")

# User inputs for network size
input_size = st.number_input("Number of Input Neurons", min_value=1, max_value=10, value=3, step=1)
hidden_size = st.number_input("Number of Hidden Neurons", min_value=1, max_value=10, value=4, step=1)
output_size = st.number_input("Number of Output Neurons", min_value=1, max_value=10, value=1, step=1)

# Create the neural network
nn = NeuralNetwork(input_size, hidden_size, output_size)

st.write("### Initial Weights")
st.write("Weights Input-Hidden:")
st.write(nn.weights_input_hidden)
st.write("Weights Hidden-Output:")
st.write(nn.weights_hidden_output)

# User inputs for training
input_vector = st.text_input(f"Input Vector (comma-separated, length={input_size})", "0.5,0.1,0.2")
target_vector = st.text_input(f"Target Output Vector (comma-separated, length={output_size})", "1")

def parse_vector(text, expected_length):
    try:
        vec = [float(x.strip()) for x in text.split(",")]
        if len(vec) != expected_length:
            st.error(f"Input vector length must be {expected_length}")
            return None
        return np.array(vec)
    except:
        st.error("Invalid input vector format")
        return None

input_data = parse_vector(input_vector, input_size)
target_data = parse_vector(target_vector, output_size)

if input_data is not None and target_data is not None:
    st.write("### Forward Propagation")
    output = nn.forward(input_data)
    st.write(f"Network Output: {output}")

    learning_rate = st.slider("Learning Rate", 0.01, 1.0, 0.1)

    if st.button("Train One Step"):
        nn.train(input_data, target_data, learning_rate)
        st.write("### Updated Weights After Training Step")
        st.write("Weights Input-Hidden:")
        st.write(nn.weights_input_hidden)
        st.write("Weights Hidden-Output:")
        st.write(nn.weights_hidden_output)

