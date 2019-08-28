######################
# Week 1
# Housing prices

"""
In this exercise you'll try to build a neural network that predicts the price of a house according to a simple formula.

So, imagine if house pricing was as easy as a house costs 50k + 50k per bedroom, 
so that a 1 bedroom house costs 100k, a 2 bedroom house costs 150k etc.

How would you create a neural network that learns this relationship 
so that it would predict a 7 bedroom house as costing close to 400k etc.

Hint: Your network might work better if you scale the house price down. 
You don't have to give the answer 400...it might be better to create something that predicts the number 4, 
and then your answer is in the 'hundreds of thousands' etc.

"""

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

xs = np.array([1,  2, 3, 4, 5], dtype=float)
ys = np.array([1,  1.5, 2, 2.5, 3], dtype=float)

# Training the Neural Network
model.fit(xs, ys, epochs=500)

# Use the model
print(model.predict([7.0]))