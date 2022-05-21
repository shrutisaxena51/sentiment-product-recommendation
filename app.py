
from flask import Flask, request
import numpy as np
import pandas as pd
from flask import app
from flask import jsonify
from functools import reduce
from flask import render_template
from model import *
import joblib

# Flask object 
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    # Get values from browser
    if (request.method == 'POST'):
        user_name = request.get_data().decode('utf-8')
        if user_name is not None:
            user_name = user_name.split('=')[1]
        print(user_name)

        top_5 = recommendation(user_name)
        recommend_prod_list=[]

        for product in top_5:
            recommend_prod_list.append(product)
        print(recommend_prod_list)

        return render_template('index.html', recommendation='Recommendations for username {} are  {}'.format(user_name,recommend_prod_list))
    else:
        return render_template('index.html')

if __name__ == "__main__":
    # Start Application
    app.run()
