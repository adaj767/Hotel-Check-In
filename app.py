import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    
    #int_features = [int(x) for x in request.form.values()]
    if request.form['Age'] <= 0 | request.form['days_creation']<=0 | request.form['avg_lead_time']<=0 | request.form['lodging_revenue']<=0 | \
     request.form['other_revenue']<=0 | request.form['persons_nights']<=0 | request.form['high_floor']<=0 | request.form['king_bed']<=0 | request.form['twin_bed']<=0:
        raise ValueError("Please enter number greater than 0")
    else:
        int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    if prediction == 0:
        return render_template('index.html', prediction_text="The customer will not be checking in")
    else:
        return render_template('index.html', prediction_text="The customer will be checking in")



if __name__ == "__main__":
    app.run(debug=True)