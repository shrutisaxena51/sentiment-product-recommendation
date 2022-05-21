import numpy as np
from flask import render_template
import pandas as pd
from flask import app
from functools import reduce
import joblib

def recommendation(username):
    # loading models
    xgb_model = joblib.load('model/xgboost_model.pkl')
    item_rating = joblib.load('model/item_based_final_rating')
    product_tfidf = joblib.load('model/product_tf_idf')

    top_list = item_rating.loc[username].sort_values(ascending=False)[0:20]
    top_20_product = top_list.index.tolist()
    sentiment_dictionary={}
    for i in top_20_product:
        reviews = product_tfidf[i]
        sentiment = [xgb_model.predict(r) for r in reviews]
        pos_sentence = 100-((reduce(lambda x,y:x+y,sentiment)/len(sentiment))*100)
        sentiment_dictionary[i] = pos_sentence
    
    top_5_product = [key for key,value in sorted(sentiment_dictionary.items(), key = lambda x:x[1], reverse=True)[:5]]

    return top_5_product


