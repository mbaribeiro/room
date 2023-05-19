import tensorflow as tf

#create a class to store all the parameters
class Parameters:
    def __init__(self):
        self.optiizer1 = tf.keras.optimizers.Adam(learning_rate=0.01)
        self.optiizer2 = tf.keras.optimizers.Adam(learning_rate=0.005)
        self.v1 = None
        self.v2 = None
        self.batch_sizeTemp = 2
        self.batch_sizeVel = 2
        self.epochsTemp = 50
        self.epochsVel = 50

    def opt(self):
        return self.optiizer1, self.optiizer2
    
    def batch_size(self):
        return self.batch_sizeTemp, self.batch_sizeVel
    
    def epochs(self):
        return self.epochsTemp, self.epochsVel