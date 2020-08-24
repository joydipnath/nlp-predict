from flask import request, jsonify
# import tensorflow as tf
import numpy as np
from keras import backend
from keras.models import load_model
from keras.preprocessing import sequence
from keras.datasets import imdb
import os
import json


class SentimentAnalysis:

    def __init__(self):
        pass

    def sentiment(self, text):
        seq_length = 300
        model = load_model('model/sentiment.h5')
        # index = json.loads('model/imdb_word_index.json')
        index = imdb.get_word_index()
        review = text
        words = review.split()
        review = []
        for word in words:
            if word not in index:
                review.append(2)
            else:
                review.append(index[word] + 3)
        review = sequence.pad_sequences([review], truncating='pre', padding='pre', maxlen=seq_length)

        prediction = model.predict(review)
        # print("Prediction (0 = negative, 1 = positive) = ", end="")
        # print("%0.4f" % prediction[0][0])
        return prediction[0][0]


    def load_model_to_app(self):
        model = load_model('sentiment.h5')
        return model
