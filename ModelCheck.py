import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, accuracy_score

# Step 1: Load the Data
df = pd.read_csv('Dataset.csv')
X = df['URL']
y = df['status']

# Step 2: Feature Extraction
vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(X)

# Step 3: Train the Model
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Train a Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 4: Calibrate the Model
# Initialize CalibratedClassifierCV without 'base_estimator'
calibrated_model = CalibratedClassifierCV(model, method='sigmoid', cv='prefit')
calibrated_model.fit(X_train, y_train)

# Step 5: Evaluate the Model
y_pred = calibrated_model.predict(X_test)
y_prob = calibrated_model.predict_proba(X_test)[:, 1]

print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Accuracy Score:")
print(accuracy_score(y_test, y_pred))

print("Sample Probabilities:")
for url, prob in zip(X_test[:5], y_prob[:5]):
    print(f"URL: {vectorizer.inverse_transform(url)[0]} - Probability of Phishing: {prob:.2f}")
