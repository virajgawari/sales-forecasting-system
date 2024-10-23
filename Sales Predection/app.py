from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

model = joblib.load('random_forest_model.pkl')

app = Flask(__name__)

label_encoders = {
    'Outlet_Size': LabelEncoder(),
    'Outlet_Location_Type': LabelEncoder(),
    'Outlet_Type': LabelEncoder(),
}

label_encoders['Outlet_Size'].fit(['Small', 'Medium', 'Large'])
label_encoders['Outlet_Location_Type'].fit(['Tier 1', 'Tier 2', 'Tier 3'])
label_encoders['Outlet_Type'].fit(['Supermarket Type1', 'Supermarket Type2', 'Grocery Store', 'Others'])

@app.route('/')
def home():
    return render_template('index.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    outlet_establishment_year = int(request.form['outlet_establishment_year'])
    outlet_size = request.form['outlet_size']
    outlet_location_type = request.form['outlet_location_type']
    outlet_type = request.form['outlet_type']

    # Prepare input data for prediction
    input_data = pd.DataFrame({
        'Outlet_Size': [label_encoders['Outlet_Size'].transform([outlet_size])[0]],
        'Outlet_Location_Type': [label_encoders['Outlet_Location_Type'].transform([outlet_location_type])[0]],
        'Outlet_Type': [label_encoders['Outlet_Type'].transform([outlet_type])[0]],
        'Outlet_Age': [2024 - outlet_establishment_year]  # Outlet Age
    })

    # Make prediction
    prediction = model.predict(input_data)

    # Return prediction as JSON
    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
