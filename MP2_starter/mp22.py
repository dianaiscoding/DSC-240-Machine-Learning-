import numpy as np
import pandas as pd
#I has Chatgbt help make code corrections and suggest formatting

def sigmoid(z):
    # Clip z to avoid overflow in exp
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))

def train_sgd(X, y, epochs=50, lr=0.01, reg=0.0, seed=0):
    """
    Training linear classifier using Stochastic Gradient Descent
    Args:
        X (np.array): Feature matrix (N_samples, N_features)
        y (np.array): Target vector (N_samples,)
        epochs (int): Number of passes over the dataset
        lr (float): Learning rate
        reg (float): L2 regularization strength (0.0 disables)
        seed (int): Random seed for reproducible shuffling
    Returns:
        w (np.array): Weights
        b (float): Bias
    """
    n_samples, n_features = X.shape

    # Initialize weights and bias to zeros
    w = np.zeros(n_features, dtype=float)
    b = 0.0

    rng = np.random.default_rng(seed)

    for epoch in range(epochs):
        # Shuffle the data at the start of each epoch
        indices = np.arange(n_samples)
        rng.shuffle(indices)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        for i in range(n_samples):
            # Forward pass (Prediction)
            linear_pred = np.dot(X_shuffled[i], w) + b
            y_pred = sigmoid(linear_pred)

            # Compute Error (y - y_hat)
            error = y_shuffled[i] - y_pred

            # Update weights and bias:
            # w = w + lr * (error * x - reg * w)
            w += lr * (error * X_shuffled[i] - reg * w)
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
    X_train = training_data.iloc[:, :-1].to_numpy(dtype=float)   # All columns except last
    y_train = training_data.iloc[:, -1].to_numpy()               # Last column (Target)

    # Harden labels: ensure y is 0/1 (in case it's -1/+1)
    unique = set(np.unique(y_train))
    if unique == {-1, 1}:
        y_train = (y_train == 1).astype(int)
    else:
        y_train = y_train.astype(int)

    if testing_data.shape[1] == training_data.shape[1]:
        X_test = testing_data.iloc[:, :-1].to_numpy(dtype=float)
    else:
        X_test = testing_data.to_numpy(dtype=float)

    # Standardize features
    # Use training mean/std only
    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0)
    sigma[sigma == 0] = 1.0

    X_train = (X_train - mu) / sigma
    X_test = (X_test - mu) / sigma

    # Training
    w, b = train_sgd(X_train, y_train, epochs=50, lr=0.05, reg=1e-4, seed=0)

    # Prediction
    predictions = []
    for i in range(len(X_test)):
        linear_out = np.dot(X_test[i], w) + b
        prob = sigmoid(linear_out)

        # Threshold probability at 0.5 to decide class 0 or 1
        if prob >= 0.5:
            predictions.append(1)
        else:
            predictions.append(0)

    return predictions
