import numpy as np
import pandas as pd
import tensorflow as tf
import ObModel as om
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "obesity_model.keras")

NN_model = tf.keras.models.load_model(model_path)

def prediction(features):
    return om.predict(features=features ,model=NN_model)

def suggestion(features):
    return om.suggestions(person=features)