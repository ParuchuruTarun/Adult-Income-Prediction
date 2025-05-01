# app.py

from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
from collections import Counter

# Load the trained model
DTmodel_path = 'Models/DTmodel.pkl'
with open(DTmodel_path, 'rb') as file:
    DTmodel = pickle.load(file)

KNNmodel_path = 'Models/KNNmodel.pkl'
with open(KNNmodel_path, 'rb') as file:
    KNNmodel = pickle.load(file)

LRmodel_path = 'Models/LRmodel.pkl'
with open(LRmodel_path, 'rb') as file:
    LRmodel = pickle.load(file)

RFmodel_path = 'Models/RFmodel.pkl'
with open(RFmodel_path, 'rb') as file:
    RFmodel = pickle.load(file)

SVMmodel_path = 'Models/SVMmodel.pkl'
with open(SVMmodel_path, 'rb') as file:
    SVMmodel = pickle.load(file)


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract data from form
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    
    # Get predictions from all models
    preds = [
        DTmodel.predict(final_features)[0],
        KNNmodel.predict(final_features)[0],
        LRmodel.predict(final_features)[0],
        RFmodel.predict(final_features)[0],
        SVMmodel.predict(final_features)[0]
    ]
    
    # Perform majority voting
    vote_result = Counter(preds).most_common(1)[0][0]
    output = 'Less then 50K' if vote_result == 1 else 'More then or equal to 50K'

    return render_template('index.html', prediction_text='Prediction: {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)