import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import optuna
from math import *
from utils.parameters import Parameters as prt
from callbacks.printCallback import PrintEpochCallback as prtC
from dataFunctions.readData import ReadData as rd
from models.neural_networks import neuralNetworks as nn
from visualisation.structureSave import structureSave as ss
from visualisation.graphs import Graphs as gr
from utils.saveModels import saveModels as sm

fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
epoch_callback_temp = prtC(axs[0])
epoch_callback_vel = prtC(axs[1])

# Separate Temperature and Velocity data
data = rd().readInput()
_inputs = data[:, :3]

_outputs1, _outputs2, coordinates = rd().trainData()

# Remove the corresponding columns in coordinates that have all zeros in both outputs
coordinates = np.array(coordinates)[~np.all(
    np.array(_outputs1, dtype=float) == 0, axis=0)]
_outputs1 = np.array(_outputs1)[:, ~np.all(
    np.array(_outputs1, dtype=float) == 0, axis=0)].astype(float)
_outputs2 = np.array(_outputs2)[:, ~np.all(
    np.array(_outputs2, dtype=float) == 0, axis=0)].astype(float)

# Normalize the inputs, outputs1 and outputs2 using min-max normalization
inputs = (_inputs - _inputs.min(axis=0)) / \
    (_inputs.max(axis=0) - _inputs.min(axis=0))
outputs1 = (_outputs1 - _outputs1.min(axis=0)) / \
    (_outputs1.max(axis=0) - _outputs1.min(axis=0))
outputs2 = (_outputs2 - _outputs2.min(axis=0)) / \
    (_outputs2.max(axis=0) - _outputs2.min(axis=0))

# Create the model
neural_net = nn()
neural_net.inputs, neural_net.outputs1, neural_net.outputs2 = inputs, outputs1, outputs2
modelTemp = neural_net.createModelTemp()
modelVel = neural_net.createModelVel()

# Save the structure
structure_save = ss()
structure_save.modelTemp, structure_save.modelVel = modelTemp, modelVel
structure_save.save()

# Test the model
predicted_inputs = rd().readInputsPred()

def objective_temp(trial, inputs, outputs1, _inputs, _outputs1, predicted_inputs):
    # Define the parameters of the optimizer
    parameters = prt()
    parameters.trial = trial
    optimizer1 = parameters.opt_temp()

    # Compile the model
    parameters.trial = trial
    modelTemp.compile(loss='mean_squared_error', optimizer=optimizer1)

    # Train the modelTemp
    modelTemp.fit(inputs, outputs1, epochs=parameters.opt_epochs()
                  [0], batch_size=parameters.opt_batch_size()[0])

    test_inputs = np.array((predicted_inputs - _inputs.min(axis=0)) / \
        (_inputs.max(axis=0) - _inputs.min(axis=0)))
    
    # Evaluate the performance of the model
    readOutputs = rd()
    readOutputs.predInputs = predicted_inputs

    #Normalize the test_outputs
    test_outputs = readOutputs.readOutputsTemp()
    test_outputs = np.array((test_outputs - _outputs1.min(axis=0)) / \
        (_outputs1.max(axis=0) - _outputs1.min(axis=0)))
    accuracy = modelTemp.evaluate(test_inputs, test_outputs, verbose=1)

    return accuracy


def objective_vel(trial, inputs, outputs2, _inputs, _outputs2, predicted_inputs):
    # Define the parameters of the optimizer
    parameters = prt()
    parameters.trial = trial
    optimizer2 = parameters.opt_vel()

    # Compile the model
    modelVel.compile(loss='mean_squared_error', optimizer=optimizer2)

    # Train the modelVel
    modelVel.fit(inputs, outputs2, epochs=parameters.opt_epochs()
                 [1], batch_size=parameters.opt_batch_size()[1])

    test_inputs = np.array((predicted_inputs - _inputs.min(axis=0)) / \
        (_inputs.max(axis=0) - _inputs.min(axis=0)))

    # Evaluate the performance of the model
    readOutputs = rd()
    readOutputs.predInputs = predicted_inputs

    # Normalize the test_outputs
    test_outputs = readOutputs.readOutputsVel()
    test_outputs = np.array((test_outputs - _outputs2.min(axis=0)) / \
        (_outputs2.max(axis=0) - _outputs2.min(axis=0)))
    accuracy = modelVel.evaluate(test_inputs, test_outputs, verbose=1)

    return accuracy

# Define the callback function to track the progress
def callbackTemp(study, trial):
    print('Trial Number:', trial.number)
    print('Current Parameters:', trial.params)
    print('Best Value:', study.best_value)
    print('----------------------------------')
    # Save in a file the study_temp
    study_temp.trials_dataframe().to_csv('study_temp.csv')

# Define the callback function to track the progress
def callbackVel(study, trial):
    print('Trial Number:', trial.number)
    print('Current Parameters:', trial.params)
    print('Best Value:', study.best_value)
    print('----------------------------------')
    # Save in a file the study_vel
    study_vel.trials_dataframe().to_csv('study_vel.csv')

study_temp = optuna.create_study()
study_temp.optimize(lambda trial: objective_temp(
    trial, inputs, outputs1, _inputs, _outputs1, predicted_inputs), n_trials=100, callbacks=[callbackTemp])

study_vel = optuna.create_study()
study_vel.optimize(lambda trial: objective_vel(
    trial, inputs, outputs2, _inputs, _outputs2, predicted_inputs), n_trials=100, callbacks=[callbackVel])

# Normalize the inputs using min-max normalization
predicted_inputs = (predicted_inputs - _inputs.min(axis=0)) / (_inputs.max(axis=0) - _inputs.min(axis=0))
test_inputs = np.array(predicted_inputs)
predicted_outputsTemp = modelTemp.predict(test_inputs)
predicted_outputsVel = modelVel.predict(test_inputs)

# Denormalize the outputs using min-max denormalization
predicted_inputs = test_inputs * (_inputs.max(axis=0) - _inputs.min(axis=0)) + _inputs.min(axis=0)
predicted_outputsTemp = predicted_outputsTemp * (_outputs1.max(axis=0) - _outputs1.min(axis=0)) + _outputs1.min(axis=0)
predicted_outputsVel = predicted_outputsVel * (_outputs2.max(axis=0) - _outputs2.min(axis=0)) + _outputs2.min(axis=0)

gr = gr()
gr.coord, gr.predInputs, gr.predOutputs_Temp, gr.predOutputs_Vel = coordinates, predicted_inputs, predicted_outputsTemp, predicted_outputsVel
gr.results()
gr.r2()

sm = sm()
sm.modelTemp, sm.modelVel = modelTemp, modelVel
sm.saveModels()