from flask import Flask, make_response, request, render_template
import io
import os
import csv
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))

@app.route('/', methods=['POST','GET'])
def hello():
    return render_template('bigmart.html')

@app.route('/predict', methods=['POST','GET'])
def predict():
    pass
    return render_template('bigmart.html',pred='Prediction of sales is calculating')

if __name__ == "__main__":
    app.run()