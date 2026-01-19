"""from flask import Flask

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello, Flask!'

if __name__ == '__main__':
    app.run(debug=True)"""



import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

with open(r'C:\Users\JASH\Desktop\MCA\Jupyter Notebook Prec\Diabetes\Diabetes_flask\venv\Diabetes_model.pkl', 'rb') as d_model:
    model = pickle.load(d_model)

with open(r'C:\Users\JASH\Desktop\MCA\Jupyter Notebook Prec\Diabetes\Diabetes_flask\venv\scaler.pkl', 'rb') as sc_file:
    scaler = pickle.load(sc_file)


@app.route('/')
def home():
    return render_template('form.html')


@app.route('/predict',methods=['GET','POST'])
def predict(): 
    try:
        #get the json data from api request
        #data = request.get_json()

        if request.method == 'POST':
            data = dict(zip(['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                                'Insulin','BMI', 'DiabetesPedigreeFunction', 'Age'],
                                [int(request.form.get('Pregnancies')),int(request.form.get('Glucose')),
                                int(request.form.get('BloodPressure')),int(request.form.get('SkinThickness')),
                                int(request.form.get('Insulin')),float(request.form.get('BMI')),
                                float(request.form.get('DiabetesPedigreeFunction')),int(request.form.get('Age'))]))
        
        inpur_data = pd.DataFrame([data])

        if not data:
            return render_template('error.html',er = "Input data not provided")
            #return jsonify({"error":"Input data not provided"}), 400

        required_columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                            'Insulin','BMI', 'DiabetesPedigreeFunction', 'Age']

        if not all(col in inpur_data.columns for col in required_columns):
            return render_template('error.html',er = f"Required columns is missing {required_columns}")
            #return jsonify({"error":f"Required columns is missing {required_columns}"}), 400

        scaled_data = scaler.transform(inpur_data)

        prediction = model.predict(scaled_data)

        response = {
            "prediction" : "Diabetes" if prediction[0] == 1 else "Not Diabetes"
        }
        return render_template('result.html',prediction = response['prediction'])

    except Exception as e:
        return render_template('error.html',er = str(e))
        #return jsonify({'error': str(e)}), 500 

if __name__ == "__main__":
    app.run(debug=True)
