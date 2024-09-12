from flask import Flask, request, render_template, jsonify
import pickle
import logging
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the trained model, vectorizer, and scaler
try:
    logger.info("Loading model, vectorizer, and scaler...")
    with open('model/logistic_regression_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('model/tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    with open('model/scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
except Exception as e:
    logger.error(f"Error loading model, vectorizer, or scaler: {e}")
    raise

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the URL from the form
        url = request.form.get('url', '').strip()

        if not url:
            return jsonify({'error': 'No URL provided'}), 400

        # Transform the URL using the loaded vectorizer
        url_tfidf = vectorizer.transform([url])

        # Check if there are additional numerical features
        # Adjust the number of numerical features based on model expectations
        num_features = 48  # Adjust this to match the actual number of features your model expects
        if url_tfidf.shape[1] < num_features:
            # Dummy numerical features (zeros or some default value)
            numerical_features = np.zeros((1, num_features - url_tfidf.shape[1]))
        else:
            numerical_features = np.zeros((1, 0))

        # Scale the numerical features
        if numerical_features.shape[1] > 0:
            scaled_numerical_features = scaler.transform(numerical_features)
            combined_features = np.hstack((url_tfidf.toarray(), scaled_numerical_features))
        else:
            combined_features = url_tfidf.toarray()

        # Ensure combined_features has the correct number of features
        expected_features = 63121  # Replace with the number of features your model expects
        if combined_features.shape[1] != expected_features:
            raise ValueError(f"Feature dimension mismatch: expected {expected_features} but got {combined_features.shape[1]}")

        # Make a prediction
        prediction = model.predict(combined_features)
        prediction_proba = model.predict_proba(combined_features)[:, 1]

        # Interpret the prediction
        result = "Phishing" if prediction[0] == 1 else "Not Phishing"
        confidence = prediction_proba[0] * 100  # Convert to percentage

        return render_template('result.html', result=result, confidence=confidence)

    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return jsonify({'error': 'Error making prediction'}), 500

if __name__ == '__main__':
    app.run(debug=True)
