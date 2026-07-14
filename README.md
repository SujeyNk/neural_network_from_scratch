# Neural Network from Scratch Using NumPy

A fully connected neural network implemented from scratch using NumPy and trained to classify handwritten digits from the scikit-learn Digits dataset.

The core model does not use TensorFlow, Keras, PyTorch autograd, or scikit-learn classifiers. Forward propagation, softmax, cross-entropy loss, backpropagation, gradient descent, mini-batch training, and L2 regularization are implemented manually.

## Project Overview

The project solves a 10-class image-classification problem. Each input is an 8 × 8 grayscale digit image flattened into 64 numerical features, and the model predicts one of the classes from 0 to 9.

### Dataset

- Source: `sklearn.datasets.load_digits`
- Samples: 1,797
- Input features: 64 pixel values per image
- Classes: 10 digits, from 0 to 9
- Original pixel range: 0 to 16
- Normalized pixel range: 0 to 1
- Train/test split: 80% / 20%

## Neural Network Architecture

Input layer:   64 features
      ↓
Hidden layer:  128 neurons + ReLU
      ↓
Output layer:  10 neurons + SoftMax


The model parameters are:

- W1: weights between the input and hidden layers
- b1: hidden-layer bias
- W2: weights between the hidden and output layers
- b2: output-layer bias

Weights are initialized from a small normal distribution and biases are initialized to zero.

## Concepts Implemented

- Feature normalization
- One-hot encoding
- Random train/test split
- Manual weight and bias initialization
- ReLU activation
- Numerically stable softmax
- Multiclass cross-entropy loss
- Forward propagation
- Backpropagation using the chain rule
- Mini-batch gradient descent
- Data shuffling before each epoch
- L2 weight regularization
- Accuracy calculation
- Training-loss visualization

## Forward Propagation

For a mini-batch X:

Z1 = XW1 + b1
H  = ReLU(Z1)
Z2 = HW2 + b2
P  = Softmax(Z2)


P contains the predicted probability distribution over the ten digit classes.

## Loss Function

The model uses multiclass cross-entropy:


Loss = -mean(sum(Y * log(P)))


A small value is added inside the logarithm to avoid taking `log(0)`.

## Backpropagation

The output gradient simplifies to:


dZ2 = (P - Y) / batch_size


The remaining gradients are calculated manually:


dW2 = H.T @ dZ2
db2 = sum(dZ2)
dH  = dZ2 @ W2.T
dZ1 = dH * ReLU'(Z1)
dW1 = X.T @ dZ1
db1 = sum(dZ1)


Parameters are updated using gradient descent:


W = W - learning_rate * dW
b = b - learning_rate * db


For L2 regularization, `lambda * W` is added to each weight gradient.

## Hyperparameters Used

| Hyperparameter | Value |
|---|---:|
| Hidden neurons | 128 |
| Learning rate | 0.1 |
| Batch size | 64 |
| Epochs | 30 |
| L2 strength | 0.0001 |
| Random seed | 0 |

## Result

The notebook output reaches approximately **95–96% test accuracy** on the Digits dataset after 30 epochs. Exact results can vary if the data split, initialization, or training procedure is changed.

## Project Structure


.
├── neural_network_from_scratch.ipynb
├── utils_train.py                 # Optional PyTorch training helper; not used by the NumPy model
├── requirements.txt
├── README.md
└── .gitignore



## Installation

Clone the repository and move into the project directory:


git clone <your-repository-url>
cd <repository-name>


Create and activate a virtual environment.

### Windows


python -m venv .venv
.venv\Scripts\activate


### macOS/Linux


python3 -m venv .venv
source .venv/bin/activate

Install the dependencies:


pip install -r requirements.txt


Start Jupyter Notebook:


jupyter notebook


Open neural_network_from_scratch.ipynb and run the cells in order.

## Important Experimental Note

For a strict comparison between training with and without L2 regularization, both experiments should start from identical freshly initialized weights. Epoch loss should also be calculated as the average loss across all mini-batches rather than using only the final mini-batch loss.

## Skills Demonstrated

- Understanding of neural-network mathematics
- Matrix-based implementation using NumPy
- Manual derivation and implementation of gradients
- Multiclass classification
- Regularization and optimization
- Model evaluation and visualization

## License

This project is intended for learning, portfolio demonstration and interview preparation.
