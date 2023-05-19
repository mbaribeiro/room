import tensorflow as tf
from math import *

class neuralNetworks():
    def __init__(self):
        self.inputs = None
        self.outputs1 = None
        self.outputs2 = None
    
    def createModelTemp(self):
        # Create the model
        modelTemp = tf.keras.Sequential([
            tf.keras.layers.Dense(round(len(self.outputs1[0])/exp(4*len(self.inputs[0]))), input_shape=(len(self.inputs[0]),)),
            tf.keras.layers.Dense(round(len(self.outputs1[0])/exp(3.5*len(self.inputs[0]))), activation="relu",),
            tf.keras.layers.Dense(round(len(self.outputs1[0])/exp(2*len(self.inputs[0]))), activation="relu"),
            tf.keras.layers.Dense(round(len(self.outputs1[0])/exp(1*len(self.inputs[0]))), activation="relu"),
            tf.keras.layers.Dense(len(self.outputs1[0]), activation='linear')
        ])
        return modelTemp

    def createModelVel(self):
        modelVel = tf.keras.Sequential([
            tf.keras.layers.Dense(round(len(self.outputs1[0])/exp(2*len(self.inputs[0]))), input_shape=(len(self.inputs[0]),)),
            tf.keras.layers.Dense(round(len(self.outputs1[0])/exp(1*len(self.inputs[0]))), activation="relu"),
            tf.keras.layers.Dense(len(self.outputs2[0]), activation='linear')
        ])
        return modelVel