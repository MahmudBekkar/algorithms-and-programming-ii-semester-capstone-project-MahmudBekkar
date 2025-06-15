import matplotlib.pyplot as plt
import streamlit as st

def plot_loss_curve(losses):
    fig, ax = plt.subplots()
    ax.plot(losses)
    ax.set_title("Loss over Epochs")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    st.pyplot(fig)

