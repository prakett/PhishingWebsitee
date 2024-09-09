import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
import pickle

# Load the dataset
print("Loading dataset...")
df_combined = pd.read_csv('Dataset.csv')
print("Dataset loaded successfully.")

# Check for missing values in 'URL' and 'status' columns
if df_combined['URL'].isnull().sum() > 0 or df_combined['status'].isnull().sum() > 0:
    raise ValueError("Missing values detected in 'URL' or 'status' columns.")

# Extract features (URLs) and labels (status)
X = df_combined['URL']  # URLs as the feature
y = df_combined['status']  # Status (0 for non-phishing, 1 for phishing)

# Split the dataset into training and testing sets
print("Splitting data into training and test sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Training data size: {X_train.shape[0]}")
print(f"Test data size: {X_test.shape[0]}")

# Text preprocessing using TF-IDF (max_features set to handle memory better)
print("Applying TF-IDF Vectorization...")
vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)  # Adjust max_features for memory usage
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
print("TF-IDF vectorization complete.")

# Train a Logistic Regression model
print("Training Logistic Regression model...")
model = LogisticRegression(solver='lbfgs', max_iter=1000)
model.fit(X_train_tfidf, y_train)
print("Model training complete.")

# Save the model and vectorizer
def save_model(model, vectorizer, model_path='logistic_regression_model.pkl', vectorizer_path='tfidf_vectorizer.pkl'):
    with open(model_path, 'wb') as model_file:
        pickle.dump(model, model_file)
    with open(vectorizer_path, 'wb') as vectorizer_file:
        pickle.dump(vectorizer, vectorizer_file)
    print(f"Model saved to {model_path} and vectorizer saved to {vectorizer_path}.")

# Evaluate the model on the test data
print("Evaluating the model...")
y_pred = model.predict(X_test_tfidf)
y_proba = model.predict_proba(X_test_tfidf)[:, 1]

# Print evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)
print(f"Accuracy score: {accuracy}")
print(f"ROC-AUC score: {roc_auc}")
print("Classification report:")
print(classification_report(y_test, y_pred))

# Save the model and vectorizer
save_model(model, vectorizer)
