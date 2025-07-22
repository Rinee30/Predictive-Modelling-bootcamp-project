from flask import Flask, render_template, request
import pickle
import numpy as np

# Flask setup
app = Flask(_name_, template_folder='my_html_pages')

# Load trained model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1. Get original 12 inputs from form
        data = [float(x) for x in request.form.values()]
        
        # Unpack necessary values
        age = data[0]
        platelets = data[6]
        serum_creatinine = data[7]

        # 2. Compute additional features
        creatinine_per_age = serum_creatinine / age
        platelet_ratio = platelets / serum_creatinine

        # 3. Combine all 14 features
        final_input = np.array([[*data, creatinine_per_age, platelet_ratio]])

        # 4. Predict
        prediction = model.predict(final_input)[0]
        result = "High Risk of Heart Failure" if prediction == 1 else "Low Risk of Heart Failure"

        return render_template('index.html', prediction_text=f"Prediction: {result}")
    
    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if _name_ == '_main_':
    app.run(debug=True)