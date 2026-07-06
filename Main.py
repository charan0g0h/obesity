import numpy as np
import pandas as pd
import tensorflow as tf
import ObModel as om

NN_model = tf.keras.models.load_model("obesity_model.keras")

def prediction(features):
    return om.predict(features=features ,model=NN_model)

def suggestion(features):
    return om.suggestions(person=features)