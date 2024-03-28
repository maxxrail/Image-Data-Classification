# Assignment 3: Classification of Image Data

## Overview
This project is designed to classify sign language gestures using a dataset formatted similarly to the classic MNIST dataset, but featuring hand sign representations for letters. It explores the efficacy of Multi-Layer Perceptron (MLP) models under various configurations and compares their performance with that of a Convolutional Neural Network (ConvNet) model.

## Installation
To run this project, you will need Python and the following libraries:
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- PyTorch (for ConvNet experiment)


## Data Preprocessing
The dataset is split into training and testing sets, with each set comprising images in a flattened format alongside their corresponding labels. The preprocessing steps include:
- Normalization of features by subtracting the mean and dividing by the standard deviation.
- One-hot encoding of labels to match the neural networks' output format.

## Neural Network Implementation
### Classes
- `NeuralNetLayer`: An abstract base class for network layers.
- `LinearLayer`: Implements a fully connected layer.
- `ReLULayer`: Applies the ReLU activation function.
- `SoftmaxOutputLayer`: Applies the softmax function for output layers.
- `MLP`: Builds a multi-layer perceptron using the layers mentioned above.
- `Optimizer`: An abstract base class for optimizers.
- `RegGradientDescentOptimizer`: Implements gradient descent with L2 regularization.
- `AdamOptimizer`: Implements the Adam optimization algorithm.

### Key Methods
- `forward()`: Propagates inputs forward through the network.
- `backward()`: Backpropagates gradients through the network.
- `fit()`: Trains the network with the specified optimizer, batch size, and iterations.
- `predict()`: Predicts class labels for given inputs.
- `evaluate_acc()`: Calculates prediction accuracy.

## Running Experiments
The project includes several experiments to explore:
1. The impact of varying hidden units and layers in MLPs.
2. Performance differences using various activation functions.
3. The effect of L2 regularization.
4. ConvNet model performance on the Sign Language MNIST dataset.
5. Optimization of MLP architecture and comparison with ConvNet performance.

## Usage
To execute the experiments, ensure all dependencies are installed and run the code sections sequentially. Adjust hyperparameters as necessary to explore different configurations.

- 
