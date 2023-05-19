import tensorflow as tf
import matplotlib.pyplot as plt

class PrintEpochCallback(tf.keras.callbacks.Callback):
    def __init__(self, ax):
        self.losses = []
        self.ax = ax
        self.ax.set_xlabel('Epoch')
        self.ax.set_ylabel('Loss')

    def on_epoch_end(self, epoch, logs=None):
        self.losses.append(logs['loss'])
        self.ax.plot(self.losses)
        self.ax.set_title(f"Epoch {epoch+1}/{self.params['epochs']}, loss: {logs['loss']:.4f}")
        plt.pause(0.01)
