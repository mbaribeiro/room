import tensorflow as tf

class structureSave:
    def __init__(self):
        self.modelTemp = None
        self.modelVel = None
    
    def save(self):
        # plot the model in a imagem with the neurons with colors for each layers and the ligations with yourselfs and the other neurons in the next layer
        tf.keras.utils.plot_model(self.modelTemp, to_file='./images/modelTemp_Structure.png', show_shapes=True, show_layer_names=True)
        tf.keras.utils.plot_model(self.modelVel, to_file='./images/modelVel_Structure.png', show_shapes=True, show_layer_names=True)