

# Phishing Detection Project

## Project Overview
This project is a phishing detection system designed to identify potentially harmful websites and classify them as phishing or non-phishing. Using machine learning techniques and a Flask-based web interface, users can submit URLs to be analyzed for signs of phishing. The model has been trained on a dataset of known phishing and legitimate URLs and achieves high accuracy in classification.

## Directory Structure
- **`app.py`**: The main Flask application.
- **`model/`**: Directory containing the trained model (`logistic_regression_model.pkl`) and supporting files.
- **`static/` and `templates/`**: Contain static assets and HTML templates for the web interface.
- **`requirements.txt`**: List of required Python packages.

## Features
- **Phishing Classification**: Model predicts if a URL is a phishing site.
- **Web Interface**: Simple and responsive Bootstrap-based UI for submitting URLs and viewing results.
- **Logging**: Logs each prediction request and result for auditing purposes.

## Requirements

The following dependencies are required to run the phishing detection project:

- **Python Libraries**:
  - `Flask==2.0.1`: Web framework to deploy the model as a web application.
  - `scikit-learn==1.0.2`: Machine learning library used for the logistic regression and random forest models.
  - `imblearn==0.9.0`: To handle class imbalance in datasets.
  - `pandas==1.3.3`: Data manipulation and analysis.
  - `numpy==1.21.2`: Fundamental package for scientific computing.
  - `joblib==1.0.1`: To save and load machine learning models.
  - `dill==0.3.4`: Serialization library to load dataset files.
  - `gunicorn==20.1.0` (optional, for deployment): Production server for running Flask applications.
  
- **Other**:
  - **TF-IDF Vectorizer** (stored as `tfidf_vectorizer.pkl`): Used for transforming URL features before feeding into the model.
  - **Trained Model File** (e.g., `logistic_regression_model.pkl`): The trained phishing detection model.

## Installation

1. Clone this repository to your local machine:
   ```bash
   git clone https://github.com/yourusername/phishing-detection.git
   cd phishing-detection
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Place the model file (`logistic_regression_model.pkl`) and TF-IDF vectorizer file (`tfidf_vectorizer.pkl`) in the `model/` directory.

## Usage

1. Run the Flask app:
   ```bash
   python app.py
   ```

2. Open a browser and go to `http://127.0.0.1:5000` to access the phishing detection interface.

3. Enter a URL to check for phishing, and the model will return a prediction of "Phishing" or "Non-Phishing."

## Model Training

If you wish to retrain the model with new data:
1. Ensure your dataset is formatted similarly to `Dataset.csv`.
2. Run the training script (if provided), or load the model training code from `app.py`.
3. Save the newly trained model as `logistic_regression_model.pkl` or update the `app.py` to point to your new model file.

---

## License
This project is licensed under the MIT License.

## Contact
For any questions or contributions, please contact `praketraj@gmail.com`.

--- 

This README template should cover everything needed for someone to set up, run, and understand your project. Let me know if you want to include additional instructions or details!
