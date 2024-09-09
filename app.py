from flask import Flask, request, render_template
import pickle

# Load the trained model and vectorizer
print("Loading model and vectorizer...")
with open('model/logistic_regression_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
with open('model/tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the URL from the form
    url = request.form['url']

    # Transform the URL using the loaded vectorizer
    url_tfidf = vectorizer.transform([url])

    # Make a prediction
    prediction = model.predict(url_tfidf)
    prediction_proba = model.predict_proba(url_tfidf)[:, 1]

    # Interpret the prediction
    result = "Phishing" if prediction[0] == 1 else "Not Phishing"
    confidence = prediction_proba[0] * 100  # Convert to percentage

    return render_template('result.html', result=result, confidence=confidence)

if __name__ == '__main__':
    app.run(debug=True)
