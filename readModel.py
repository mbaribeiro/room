import tensorflow as tf

# read the model
modelTemp = tf.keras.models.load_model('./modelsResults/modelTemp.h5')
modelVel = tf.keras.models.load_model('./modelsResults/modelVel.h5')