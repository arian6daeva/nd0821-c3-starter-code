# Script to train machine learning model.

# Import necessary modules
import os
import sys
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split

# Import custom functions
from ml.data import process_data
from ml.model import train_model, inference, compute_model_metrics

# Define file paths
file_dir = os.path.dirname(__file__)
data_path = os.path.join(file_dir, '../data/clean_census.csv')
model_path = os.path.join(file_dir, '../model/rf_model.pkl')
encoder_path = os.path.join(file_dir, '../model/encoder.pkl')
lb_path = os.path.join(file_dir, '../model/lb.pkl')

# Load data
data = pd.read_csv(data_path)

# Split data into training and testing sets
train_data, test_data = train_test_split(data, test_size=0.20)

# Process training data with custom function
cat_features = ["workclass", "education", "marital-status", "occupation", "relationship", "race", "sex", "native-country"]
X_train, y_train, encoder, lb = process_data(train_data, categorical_features=cat_features, label="salary", training=True)

# Process testing data with custom function
X_test, y_test, _, _ = process_data(test_data, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb)

# Train and save model
rf_model = train_model(X_train, y_train)
pickle.dump(rf_model, open(model_path, 'wb'))
pickle.dump(encoder, open(encoder_path, 'wb'))
pickle.dump(lb, open(lb_path, 'wb'))

# Make predictions on test data and compute metrics
preds = inference(rf_model, X_test)
precision, recall, fbeta = compute_model_metrics(y_test, preds)

# Print metrics
print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F-beta score: {fbeta:.4f}")
