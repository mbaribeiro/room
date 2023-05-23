import h5py
import numpy as np
import pandas as pd
from math import *
from dataFunctions.readData import ReadData as rd
from visualisation.graphs import Graphs as gr
from utils.saveModels import saveModels as sm

f1 = h5py.File('./modelsResults/modelTemp.h5', 'r')
f2 = h5py.File('./modelsResults/modelVel.h5', 'r')

# read the structure of the model with tf
import tensorflow as tf
modelTemp = tf.keras.models.load_model(f1)
modelVel = tf.keras.models.load_model(f2)
modelTemp.summary()
modelVel.summary()

# Separate Temperature and Velocity data
data = rd().readInput()
_inputs = data[:, :3]

_outputs1, _outputs2, coordenates = rd().trainData()

# Remove the corresponding columns in coordinates that have all zeros in both outputs
coordenates = np.array(coordenates)[~np.all(
    np.array(_outputs1, dtype=float) == 0, axis=0)]
_outputs1 = np.array(_outputs1)[:, ~np.all(
    np.array(_outputs1, dtype=float) == 0, axis=0)].astype(float)
_outputs2 = np.array(_outputs2)[:, ~np.all(
    np.array(_outputs2, dtype=float) == 0, axis=0)].astype(float)

# Test the model
predicted_inputs = rd().readInputsPred()

# Normalize the inputs using min-max normalization
predicted_inputs = (predicted_inputs - _inputs.min(axis=0)) / (_inputs.max(axis=0) - _inputs.min(axis=0))
test_inputs = np.array(predicted_inputs)
predicted_outputsTemp = modelTemp.predict(test_inputs)
predicted_outputsVel = modelVel.predict(test_inputs)

# Denormalize the outputs using min-max denormalization
predicted_inputs = test_inputs * (_inputs.max(axis=0) - _inputs.min(axis=0)) + _inputs.min(axis=0)
predicted_outputsTemp = predicted_outputsTemp * (_outputs1.max(axis=0) - _outputs1.min(axis=0)) + _outputs1.min(axis=0)
predicted_outputsVel = predicted_outputsVel * (_outputs2.max(axis=0) - _outputs2.min(axis=0)) + _outputs2.min(axis=0)

# Plote the bars graph on a specific cilinder coordinate
cilinder = [10, 20, 15, 5, 20, -1]
gr = gr()
gr.coord, gr.predInputs, gr.predOutputs_Temp, gr.predOutputs_Vel, gr.cilinder = coordenates, predicted_inputs, predicted_outputsTemp, predicted_outputsVel, cilinder
gr.cilinderGraph()