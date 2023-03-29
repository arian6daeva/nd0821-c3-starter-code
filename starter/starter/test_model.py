from ml.model import train_model, compute_model_metrics, inference
import os
import sys
import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

file_dir = os.path.dirname(__file__)
sys.path.insert(0, file_dir)


# Define a fixture that creates sample data for testing

@pytest.fixture(scope='module')
def sample_data():
    # Generate a random dataset for a classification problem
    X, y = make_classification(n_samples=100, n_features=20, random_state=42)
    # Split the dataset into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


# Test the train_model function to ensure model training is successful
def test_train_model(sample_data):
    X_train, _, y_train, _ = sample_data
    model = train_model(X_train, y_train)
    assert model is not None, "Model training failed."
    assert hasattr(model, "predict"), "Model should have a predict method."


# Test the compute_model_metrics function to ensure it returns valid metrics
def test_compute_model_metrics(sample_data):
    _, X_test, _, y_test = sample_data
    X_train, y_train = X_test.copy(), y_test.copy()
    model = train_model(X_train, y_train)
    preds = inference(model, X_test)

    precision, recall, fbeta = compute_model_metrics(y_test, preds)
    assert 0 <= precision <= 1, "Precision should be between 0 and 1."
    assert 0 <= recall <= 1, "Recall should be between 0 and 1."
    assert 0 <= fbeta <= 1, "F1 score should be between 0 and 1."


# Test the inference function to ensure it returns predictions with the
# correct format
def test_inference(sample_data):
    X_train, X_test, y_train, _ = sample_data
    model = train_model(X_train, y_train)
    preds = inference(model, X_test)

    assert len(preds) == len(
        X_test), "Number of predictions should match the number of test samples."
    assert set(preds).issubset({0, 1}), "Predictions should be either 0 or 1."
