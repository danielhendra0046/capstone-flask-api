import json
from flask import Flask, jsonify, request
import pandas as pd
import numpy as np
import tensorflow as tf
import data_prep


app = Flask(__name__)

employees = [
  { 'id': 1, 'name': 'Mary' },
  { 'id': 2, 'name': 'Harry' },
  { 'id': 3, 'name': 'Sally' }
]
data = pd.read_csv("Groceries_dataset.csv")

@app.route('/predict', methods=['GET'])
def predict():
  model = tf.keras.models.load_model("model.h5")
  return model

app.run()
