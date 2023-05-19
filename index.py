import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from math import *
from utils.parameters import Parameters as prt
from callbacks.printCallback import PrintEpochCallback as prtC
from dataFunctions.readData import ReadData as rd
from models.neural_networks import neuralNetworks as nn
from visualisation.structureSave import structureSave as ss
from visualisation.graphs import Graphs as gr
from utils.saveModels import saveModels as sm

# Define the parameters of the optimizer
optimizer1, optimizer2 = prt().opt()

fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
epoch_callback_temp = prtC(axs[0])
epoch_callback_vel = prtC(axs[1])

# Separate Temperature and Velocity data
data = rd().readInput()
inputs = data[:, :3]
inputs[:, 0] = inputs[:, 0] - 15  # Normalize Temperature

outputs1, outputs2, coordenates = rd().trainData()

# Remove the corresponding columns in coordinates that have all zeros in both outputs
coordenates = np.array(coordenates)[~np.all(
    np.array(outputs1, dtype=float) == 0, axis=0)]
outputs1 = np.array(outputs1)[:, ~np.all(
    np.array(outputs1, dtype=float) == 0, axis=0)].astype(float)
outputs2 = np.array(outputs2)[:, ~np.all(
    np.array(outputs2, dtype=float) == 0, axis=0)].astype(float)

# Create the model
nn = nn()
nn.inputs, nn.outputs1, nn.outputs2 = inputs, outputs1, outputs2
modelTemp = nn.createModelTemp()
modelVel = nn.createModelTemp()

# Save the structure
ss = ss()
ss.modelTemp, ss.modelVel = modelTemp, modelVel
ss.save()

# Compile the model
modelTemp.compile(loss='mean_squared_error', optimizer=optimizer1)
modelVel.compile(loss='mean_squared_error', optimizer=optimizer2)

# Train the modelTemp
modelTemp.fit(inputs, outputs1, epochs=prt().epochs()[
              0], batch_size=prt().epochs()[0], callbacks=[epoch_callback_temp])
modelVel.fit(inputs, outputs2, epochs=prt().epochs()[
             1], batch_size=prt().epochs()[1], callbacks=[epoch_callback_vel])

# Test the model
predicted_inputs = rd().readInputsPred()
test_inputs = np.array(predicted_inputs)
test_inputs[:, 0] = test_inputs[:, 0] - 15
predicted_outputsTemp = modelTemp.predict(test_inputs)
predicted_outputsVel = modelVel.predict(test_inputs)

gr = gr()
gr.coord, gr.predInputs, gr.predOutputs_Temp, gr.predOutputs_Vel = coordenates, predicted_inputs, predicted_outputsTemp, predicted_outputsVel
# gr.results()
gr.r2()

sm = sm()
sm.modelTemp, sm.modelVel = modelTemp, modelVel
sm.saveModels()
