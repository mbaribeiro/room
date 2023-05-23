import tensorflow as tf

# Create a class to store all the parameters
class Parameters:
    def __init__(self, trial=None):
        self.trial = trial
        self.learning_rate1 = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.learning_rate2 = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.batch_sizeTemp = 2
        self.batch_sizeVel = 2
        self.epochsTemp = 100
        self.epochsVel = 100

    def learning_rate(self):
        return self.learning_rate1, self.learning_rate2
    
    def batch_size(self):
        return self.batch_sizeTemp, self.batch_sizeVel
    
    def epochs(self):
        return self.epochsTemp, self.epochsVel

    def opt_temp(self):
        learning_rate = self.trial.suggest_loguniform('learning_rate', 1e-5, 1e-1)
        opt_temp = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        return opt_temp
    
    def opt_vel(self):
        learning_rate = self.trial.suggest_loguniform('learning_rate', 1e-5, 1e-1)
        opt_vel = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        return opt_vel

    def opt_batch_size(self):
        batch_size_temp = self.trial.suggest_int('batch_size_temp', 2, 9)
        batch_size_vel = self.trial.suggest_int('batch_size_vel', 2, 9)
        return batch_size_temp, batch_size_vel

    def opt_epochs(self):
        epochs_temp = self.trial.suggest_int('epochs_temp', 500, 5000)
        epochs_vel = self.trial.suggest_int('epochs_vel', 500, 5000)
        return epochs_temp, epochs_vel