import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from algorithm import SimpleNeuralNetwork
from utils import generate_data

st.title("🧠 Neural Network Visualizer")

st.sidebar.header("Training Settings")
hidden_size = st.sidebar.slider("Hidden layer size", 1, 20, 5)
learning_rate = st.sidebar.slider("Learning rate", 0.01, 1.0, 0.1)
epochs = st.sidebar.slider("Epochs", 10, 1000, 100)

X_train, X_test, y_train, y_test = generate_data()

nn = SimpleNeuralNetwork(input_size=2, hidden_size=hidden_size, output_size=1, learning_rate=learning_rate)
nn.train(X_train, y_train, epochs=epochs)

y_pred = nn.predict(X_test)
y_pred_class = (y_pred > 0.5).astype(int)

accuracy = np.mean(y_pred_class == y_test)
st.write(f"### Test Accuracy: {accuracy * 100:.2f}%")

fig, ax = plt.subplots()
scatter = ax.scatter(X_test[:, 0], X_test[:, 1], c=y_pred_class.reshape(-1), cmap="coolwarm", s=20)
ax.set_title("Test Predictions")
st.pyplot(fig)
