import pickle
import pandas as pd
from sklearn.metrics import classification_report

# Define paths according to Flask directories
data_path = 'Dataset.csv'
model_path = "model/logistic_regression_model.pkl"
vectorizer_path = 'model/tfidf_vectorizer.pkl'

# Load the existing model and vectorizer
print("Loading model and vectorizer...")
with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)
with open(vectorizer_path, 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Load the dataset
print("Loading dataset...")
data = pd.read_csv(data_path)

# Preprocess the dataset
print("Preprocessing dataset...")
X = data['URL']  # Replace with the correct column if different
y = data['status']  # Replace with the correct column if different

# Transform the data using the existing vectorizer
print("Transforming data using the existing vectorizer...")
X_transformed = vectorizer.transform(X)

# Make predictions
print("Making predictions...")
y_pred = model.predict(X_transformed)

# Evaluate the model
print("Evaluating the model...")
print(classification_report(y, y_pred))

print("Accuracy check complete.")
