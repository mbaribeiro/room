import tensorflow as tf

#create a class to store all the parameters
class Parameters:
    def __init__(self):
        self.optiizer1 = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.optiizer2 = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.batch_sizeTemp = 2
        self.batch_sizeVel = 2
        self.epochsTemp = 100
        self.epochsVel = 100

    def opt(self):
        return self.optiizer1, self.optiizer2
    
    def batch_size(self):
        return self.batch_sizeTemp, self.batch_sizeVel
    
    def epochs(self):
        return self.epochsTemp, self.epochsVel