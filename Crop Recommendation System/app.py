from flask import Flask, render_template, request
import numpy as np
import os
import joblib

# Initialize Flask app
app = Flask(__name__)

# Load the trained model

model = joblib.load('models/model.pkl')
scaler = joblib.load('models/minmax_scaler.pkl')



# --- ROUTES ---

# Main page route
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        try:
            # Get input values from form
            values = [float(request.form[f'feature{i}']) for i in range(1, 8)]
            
            # Prepare dictionary to pass back input values for form retention
            values_dict = {f'feature{i}': request.form[f'feature{i}'] for i in range(1, 8)}

            input_array = np.array([values])
            input_scaled = scaler.transform(input_array)
            prediction = model.predict(input_scaled)
            
            
            crop_dict = {1: "Rice", 2: "Maize", 3: "Chickpea", 4: "Kidneybeans", 5: "Pigeonpeas", 6: "Mothbeans", 7: "Mungbean",
                 8: "Blackgram", 9: "Lentil", 10: "Pomegranate", 11: "Banana", 12: "Mango", 13: "Grapes",
                 14: "Watermelon", 15: "Muskmelon", 16: "Apple", 17: "Orange", 18: "Papaya",
                 19: "Coconut", 20: "Cotton", 21: "Jute", 22: "Coffee"}

            if prediction[0] in crop_dict:
                crop = crop_dict[prediction[0]]
                result = "Best Choice: {}".format(crop)
            else:
                result = "Sorry, we could not determine the best crop to be cultivated with the provided data."
            return render_template('index.html',result = result, input_data = values_dict)

            # result = prediction[0]
            # return render_template('index.html', prediction=result, input_data=values_dict)
        
        except Exception as e:
            return f"Error: {e}"

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
