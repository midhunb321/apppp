from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

app = Flask(__name__, static_url_path='/static')

# Load the trained model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Function to check if input values are within specified ranges
def validate_input(age, overall_rating, positions_encoded):
    if age >= 100:
        return False, "Age must be less than 100."
    if overall_rating > 100:
        return False, "Overall rating must be 100 or below."
    if positions_encoded > 3:
        return False, "Position encoded must be 3 or below."
    return True, None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from the form
        age = int(request.form['age'])
        overall_rating = int(request.form['overall_rating'])
        positions_encoded = int(request.form['positions_encoded'])
        
        # Validate input
        valid_input, error_message = validate_input(age, overall_rating, positions_encoded)
        if not valid_input:
            return render_template('index.html', error=error_message)
        
        # Make prediction
        prediction = model.predict([[age, overall_rating, positions_encoded]])
        
        # Return prediction
        return render_template('result.html', prediction=prediction[0])
    
    except Exception as e:
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)
