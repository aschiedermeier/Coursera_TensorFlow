######################
# Week 1
# Try it for yourself

import tensorflow as tf
import numpy as np
from tensorflow import keras

# Define and Compile the Neural Network
# the simplest possible neural network. It has 1 layer, and that layer has 1 neuron, and the input shape to it is just 1 value.
model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])

# Now we compile our Neural Network. When we do so, we have to specify 2 functions, a loss and an optimizer.
# 'MEAN SQUARED ERROR' for the loss and 'STOCHASTIC GRADIENT DESCENT' for the optimizer.
model.compile(optimizer='sgd', loss='mean_squared_error')

# Providing the Data

xs = np.array([-1.0,  0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

# Training the Neural Network

model.fit(xs, ys, epochs=500)

# Use the model
print(model.predict([10.0]))