import numpy as np
import pandas as pd


def sigmoid(z):
    # Clip z to avoid overflow in exp
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def train_sgd(x, y, epochs=50, lr=0.01):
    """
    Training linear classifier using Stochastic Gradient Descent

    Args:
        x (np.array): Feature matrix (N_samples, N_features)
        y (np.array): Target vector (N_samples,)
        epochs (int): Number of passes over the dataset
        lr (float): Learning rate

    Returns:
        w (np.array): Weights
        b (float): Bias
    """
    n_samples, n_features = x.shape

    # Initialize weights and bias to zeros
    w = np.zeros(n_features)
    b = 0.0

    for epoch in range(epochs):
        # Shuffle the data at the start of each epoch
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        x_shuffled = x[indices]
        y_shuffled = y[indices]

        for i in range(n_samples):
            # Forward pass (Prediction)
            linear_pred = np.dot(x_shuffled[i], w) + b
            y_pred = sigmoid(linear_pred)

            # Compute Error (y - y_hat)
            error = y_shuffled[i] - y_pred

            # Update weights and bias:Rule: w = w + learning_rate * error * input
            w += lr * error * x_shuffled[i]
            b += lr * error

    return w, b


def run_train_test(training_data, testing_data):
    """
    Train the model and generate predictions

    Args:
        training_data (pd.DataFrame): Training set with features and labels.
        testing_data (pd.DataFrame): Testing set (features only or with labels).

    Returns:
        list: A list of predicted labels (0 or 1) for the testing_data.
    """
    # Data Preparation
    # Extract NumPy arrays from DataFrames

    x_train = training_data.iloc[:, :-1].values  # All columns except last
    y_train = training_data.iloc[:, -1].values  # Last column (Target)

    if testing_data.shape[1] == training_data.shape[1]:
        x_test = testing_data.iloc[:, :-1].values
    else:
        x_test = testing_data.values

    # Training
    # Hyperparameters: lr=0.1 and epochs=20
    w, b = train_sgd(x_train, y_train, epochs=20, lr=0.1)

    # Prediction
    predictions = []
    for i in range(len(x_test)):
        linear_out = np.dot(x_test[i], w) + b
        prob = sigmoid(linear_out)

        # Threshold probability at 0.5 to decide class 0 or 1
        if prob >= 0.5:
            predictions.append(1)
        else:
            predictions.append(0)

    return predictions
