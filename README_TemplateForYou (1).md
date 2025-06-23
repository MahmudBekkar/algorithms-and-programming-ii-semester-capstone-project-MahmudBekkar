# Neural Network Visualization - Interactive Visualization

## Project Overview

This project is an interactive web application that implements and visualizes Neural Network Training, developed as part of the Algorithms and Programming II course at Fırat University, Software Engineering Department. The tool allows users to input data, configure parameters, and observe the forward propagation and weight updates in a basic neural network.

## Algorithm Description

The goal of this project is to visually demonstrate how a simple feedforward neural network can learn to classify input data. It helps beginners understand how inputs are processed, how weights are updated through training, and how predictions are improved.

### Problem Definition


This project addresses the challenge of understanding how a simple neural network learns to classify data through training. Specifically, it focuses on solving the XOR classification problem, which is not linearly separable and cannot be solved by basic linear models.
The algorithm demonstrates how a feedforward neural network with one hidden layer can:
Accept user-defined inputs and labels.
Perform forward propagation to generate predictions.
Calculate the error between predicted and actual values.
Use backpropagation to adjust weights and biases.
Improve classification accuracy through iterative learning (epochs).
By visualizing these steps, the application helps students and beginners in artificial intelligence understand how neural networks learn complex patterns and make decisions.


### Mathematical Background

A simple feedforward neural network consists of neurons arranged in layers. Each neuron performs a weighted sum of its inputs and passes the result through an activation function.
1. Forward Propagation
   For an input vector x, weights W, and bias b, the output z of a neuron is calculated as:
    z = W · x + b

   Then, the neuron applies an activation function, typically the sigmoid
  a = 1 / (1 + e^(-z))
   This is done for each layer to compute the final output of the network


2. Loss Function
To measure how far the prediction is from the true label, we use a loss function. For binary classification, a common choice is binary cross-entropy or mean squared error:
 Loss = 0.5 * (y_true - y_pred)^2

3. Backpropagation
To minimize the loss, we update the weights using gradient descent:
 W = W - α * (∂Loss/∂W)


α is the learning rate
(∂Loss/∂W) is the gradient of the loss with respect to weights

4. Training Process
This process is repeated over several epochs, allowing the network to adjust its weights and biases gradually to reduce the loss and improve accuracy
 

### Algorithm Steps

1 Initialize weights and biases randomly.
2 For each training example, compute the output via forward propagation.
3 Calculate the loss between predicted and actual output.
4 Update weights and biases using backpropagation.
5 Repeat for a set number of epochs or until convergence.


### Pseudocode

```
initialize weights and biases
for epoch in range(max_epochs):
    for input, target in dataset:
        prediction = forward_propagation(input)
        loss = compute_loss(prediction, target)
        gradients = backpropagate(prediction, target)
        update weights and biases


```

## Complexity Analysis

### Time Complexity

- **Best Case:** O(n * m) - minimal training steps
- **Average Case:** O(n * m * epochs) -  typical learning
- **Worst Case:** O(n * m * epochs) - slow convergence
  n = number of samples, m = number of neurons

  
### Space Complexity

- O(m) -memory required to store weights, biases, gradients, and activations.

  
## Features

-Interactive parameter input (learning rate, epochs, etc.)
-Real-time visualization of training steps
-Display of prediction accuracy and loss
-Easy-to-understand graphical feedback



## Screenshots

![Main Interface](https://github.com/FiratUniversity-IJDP-SoftEng/algorithms-and-programming-ii-semester-capstone-project-MahmudBekkar/blob/d254bdc340aac57ac1d33ceabe8724426bc26741/main_interface.png)
*This screenshot shows the initial interface of the neural network visualization app.*

![Algorithm in Action](https://github.com/FiratUniversity-IJDP-SoftEng/algorithms-and-programming-ii-semester-capstone-project-MahmudBekkar/blob/1f46598c1b86929938d2c4cb368df73a07487597/2.png)
![](https://github.com/FiratUniversity-IJDP-SoftEng/algorithms-and-programming-ii-semester-capstone-project-MahmudBekkar/blob/1f46598c1b86929938d2c4cb368df73a07487597/3.png)
![](https://github.com/FiratUniversity-IJDP-SoftEng/algorithms-and-programming-ii-semester-capstone-project-MahmudBekkar/blob/1f46598c1b86929938d2c4cb368df73a07487597/4.png)


## Installation


### Prerequisites

- Python 3.8 or higher
- Git

### Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/MahmudBekkar/your-repository.git
   cd your-repository

   ```

2. Create a virtual environment:
   ```bash
   # On Windows
   python -m venv venv
   venv\Scripts\activate

   # On macOS/Linux
   python3 -m venv venv
   source venv/bin/activate

   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the application:
   ```bash
   streamlit run app.py
   ```

## Usage Guide

1 Launch the Streamlit app.
2 Input your dataset points and labels.
3 Choose training parameters: learning rate, epochs, etc.
4 Click the Train button.
5 Observe training progress and output.

### Example Inputs

Inputs: [(0, 0), (0, 1), (1, 0), (1, 1)]
Labels: [0, 1, 1, 0] (XOR)
Epochs: 1000
Learning rate: 0.1



## Implementation Details

### Key Components

- `algorithm.py`: Contains the core algorithm implementation
- `app.py`: Main Streamlit application
- `utils.py`: Helper functions for data processing
- `visualizer.py`: Functions for visualization

### Code Highlights

```python
def forward_propagation(x):
    z = np.dot(W, x) + b
    a = sigmoid(z)
    return a

def backpropagation(x, y):
    # Gradient calculation
    ...

```

## Testing

This project includes a test suite to verify the correctness of the algorithm implementation:

```bash
python -m unittest test_algorithm.py
```

### Test Cases

-XOR classification test
-Learning rate bounds test
-Empty dataset edge case test

## Live Demo

A live demo of this application is available at: [https://algorithms-and-programming-ii-semester-capstone-project-mahmud.streamlit.app/]

## Limitations and Future Improvements

### Current Limitations

-Only supports 2D binary classification
-No UI for saving models
-Only basic sigmoid activation.

### Planned Improvements

-Support for multi-class classification
-Add dropout/regularization support
-Add ReLU and tanh activation options
-Improve visualization UI with animation



## References and Resources

### Academic References

1 Ian Goodfellow, Yoshua Bengio, Aaron Courville — Deep Learning
2 Michael Nielsen — Neural Networks and Deep Learning
3 C. M. Bishop — Pattern Recognition and Machine Learning



### Online Resources

-Streamlit Documentation
-VisuAlgo - Neural Networks
-Neural Network Tutorials - GeeksforGeeks, Towards Data Science


## Author

- **Name:** [Mahmud Bekkar]
- **Student ID:** [220543602]
- **GitHub:** [MahmudBekkar]

## Acknowledgements

I would like to thank Assoc. Prof. Ferhat UÇAR for guidance throughout this project, and [any other acknowledgements].

---

*This project was developed as part of the Algorithms and Programming II course at Fırat University, Technology Faculty, Software Engineering Department.*
