from flask import Flask, request, render_template
import pickle

app = Flask(__name__, static_url_path='/static')

# Load the trained model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

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
        
        # Make prediction
        prediction = model.predict([[age, overall_rating, positions_encoded]])
        
        # Return prediction
        return render_template('result.html', prediction=prediction[0])
    
    except Exception as e:
        return str(e)

if __name__ == '__main__':
    app.run(debug=True)
