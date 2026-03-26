# Starter code for DSC 240 MP3
import random

import math
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

np.random.seed(0)

def compute_metric(labels, expected):
    tp = np.sum(labels[expected == 1])
    fp = np.sum(labels[expected == 0])
    tn = np.sum(1-labels[expected == 0])
    fn = np.sum(1-labels[expected == 1])
    tpr = tp/(tp+fn)
    fpr = fp/(fp+tn)
    error_rate = (fp+fn)/(tp+fp+tn+fn)
    accuracy = (tp+tn)/(tp+fp+tn+fn)
    precision = tp/(tp+fp)
    f1 = 2*tp/(2*tp+fp+fn)

    return {
        "f1": f1,
        "accuracy": accuracy,
        "precision": precision,
        "tpr": tpr,
        "fpr": fpr,
        "error_rate": error_rate,
    }


def run_train_test(training_data: pd.DataFrame, testing_data: pd.DataFrame) -> List[int]:
    """
    Implement the training and testing procedure here. You are permitted
    to use additional functions but DO NOT change this function definition.

    Inputs:
        training_data: 
        testing_data: the same as training_data with "target" removed.

    Output:
        testing_prediction: List[int]
    """

    # Separate features and target in training data
    target_col = "target"
    y_train = training_data[target_col].values
    X_train = training_data.drop(columns=[target_col])

    # testing_data is already passed in without target
    X_test = testing_data.copy()

    # Detect feature types based on training data only
    numeric_cols = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X_train.select_dtypes(exclude=["int64", "float64"]).columns.tolist()

    binary_cols = []
    continuous_cols = []

    for col in numeric_cols:
        uniques = X_train[col].dropna().unique()
        if set(uniques).issubset({0, 1}):
            binary_cols.append(col)
        else:
            continuous_cols.append(col)

    # Preprocessing:
    # - Continuous: Standardize (normalize)
    # - Categorical: One-hot encode
    # - Binary: passthrough (0/1 already suitable)
    transformers = []
    if continuous_cols:
        transformers.append(("num", StandardScaler(), continuous_cols))
    if categorical_cols:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols))
    if binary_cols:
        transformers.append(("bin", "passthrough", binary_cols))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

    # Random forest classifier with class_weight to handle class imbalance
    rf_clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )

    # Full pipeline: SAME preprocessing for training and testing data
    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("clf", rf_clf),
        ]
    )

    # Train model
    model.fit(X_train, y_train)

    # Predict on test data
    test_pred = model.predict(X_test)

    # Return numpy array of ints to work with compute_metric
    return test_pred.astype(int)


if __name__ == '__main__':

    training = pd.read_csv('./data/train.csv')
    development = pd.read_csv('./data/dev.csv')

    target_label = development['target']
    development.drop('target', axis=1, inplace=True)
    prediction = run_train_test(training, development)
    target_label = target_label.values
    status = compute_metric(prediction, target_label)
    print(status)

    


    


