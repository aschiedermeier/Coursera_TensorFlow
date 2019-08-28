######################
# Course 1 - Part 4 - Lesson 2
# Beyond Hello World, A Computer Vision Example
# Week 2
# Comupter Vision
# Fashiion MNIST

# https://github.com/lmoroney/dlaicourse/blob/master/Course%201%20-%20Part%204%20-%20Lesson%202%20-%20Notebook.ipynb
# https://colab.research.google.com/drive/1vuLCU891-zcYybLakGnvrXyb9UR4dpob


import tensorflow as tf
print(tf.__version__)

# The Fashion MNIST data is available directly in the tf.keras datasets API. You load it like this:
mnist = tf.keras.datasets.fashion_mnist

# Calling load_data on this object will give you two sets of two lists, 
# these will be the training and testing values for the graphics that contain the clothing items and their labels.
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()



# What does these values look like? 
# Let's print a training image, and a training label to see...
# Experiment with different indices in the array. 
# For example, also take a look at index 42...that's a a different boot than the one at index 0

import matplotlib.pyplot as plt
plt.imshow(training_images[0])
print(training_labels[0])
print(training_images[0])


# You'll notice that all of the values in the number are between 0 and 255. 
# If we are training a neural network, for various reasons it's easier if we treat all values as between 0 and 1,
#  a process called 'normalizing'...and fortunately in Python it's easy to normalize a list like this without looping. 
training_images  = training_images / 255.0
test_images = test_images / 255.0

# Design model 

# Sequence(): sequence  of layers in a neural network
# Flatten(): transform picture square into number array
# Dense(): layer with 128 neurons and relu activation function
# 128 neurons: arbitrary, the higher the better performance, but also slower
# relu: "If X>0 return X, else return 0" 
# Dense(): layer with 10 neurons and softmax activation function
# 10 neurons: Fixed, as there are 10 classes of clothes
# softmax: takes biggest value in array

model = tf.keras.models.Sequential([tf.keras.layers.Flatten(), 
                                    tf.keras.layers.Dense(128, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)]) 

# Build model
model.compile(optimizer = tf.train.AdamOptimizer(),
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train model with training data, 5 rounds
model.fit(training_images, training_labels, epochs=5)

# Evaluate model with test data
model.evaluate(test_images, test_labels)

###########################################
# EXPLORATION EXERCISES

# Classifications

# Create a set of classifications for each of the test images, and then print the first entry in the classifications. 
# The output is a list of 10 numbers, representing the probability for each of the fashion classes.
# The first value [0] has the highest value at [9] -> representing high boot

classifications = model.predict(test_images)
print(classifications[0])

# should return 9 for high boot
print(test_labels[0])