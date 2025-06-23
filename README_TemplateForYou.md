# Neural Network Visualization - Interactive Visualization

## Project Overview

This project is an interactive web application that implements and visualizes Neural Network Training, developed as part of the Algorithms and Programming II course at Fırat University, Software Engineering Department. The tool allows users to input data, configure parameters, and observe the forward propagation and weight updates in a basic neural network.

## Algorithm Description

The goal of this project is to visually demonstrate how a simple feedforward neural network can learn to classify input data. It helps beginners understand how inputs are processed, how weights are updated through training, and how predictions are improved.

### Problem Definition

[Clearly define the problem that the algorithm solves]

### Mathematical Background

[Explain any mathematical concepts, formulas, or notation relevant to understanding the algorithm]

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
- **Average Case:** O(n * m * epochs) - [average training requires multiple epochs]
- **Worst Case:** O(n * m * epochs) - [maximum epochs needed for convergence]

### Space Complexity

- O(m) - memory needed for weights, activations, and gradients

## Features

-Interactive parameter input (learning rate, epochs, etc.)
-Real-time visualization of training steps
-Display of prediction accuracy and loss
-Easy-to-understand graphical feedback



## Screenshots

![Main Interface](docs/screenshots/main_interface.png)
*This screenshot shows the initial interface of the neural network visualization app.*

![Algorithm in Action](docs/screenshots/algorithm_demo.png)
*Caption describing the algorithm in action*

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

A live demo of this application is available at: [Insert Streamlit Cloud URL here]

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
-[Neural Network Tutorials - GeeksforGeeks, Towards Data Science]


## Author

- **Name:** [Mahmud Bekkar]
- **Student ID:** [220543602]
- **GitHub:** [MahmudBekkar]

## Acknowledgements

I would like to thank Assoc. Prof. Ferhat UÇAR for guidance throughout this project, and [any other acknowledgements].

---

*This project was developed as part of the Algorithms and Programming II course at Fırat University, Technology Faculty, Software Engineering Department.*
