######################
# Week 3
# Part 6 - Lesson 2
# Improving Computer Vision Accuracy using Convolutions

# Here’s the notebook that Laurence was using in that screencast. 
# To make it work quicker, go to the ‘Runtime’ menu, and select ‘Change runtime type’. 
# Then select GPU as the hardware accelerator!
# https://colab.research.google.com/drive/1Bw1gUoscJ1BHIYtJIj1HL2-zZVwZorFw


import tensorflow as tf
print(tf.__version__)
mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

# training and test data needed to be reshaped. 
# That's because the first convolution expects a single tensor containing everything. I
# nstead of 60,000 28x28x1 items in a list, we have a single 4D list that is 60,000x28x28x1. 
# If you don't do this, you'll get an error when training as the Convolutions do not recognize the shape.
training_images=training_images.reshape(60000, 28, 28, 1)
training_images=training_images / 255.0
test_images = test_images.reshape(10000, 28, 28, 1)
test_images=test_images/255.0

# Next is to define your model. Now instead of the input layer at the top, you're going to add a Convolution. The parameters are:
# The number of convolutions you want to generate. Purely arbitrary, but good to start with something in the order of 32
# The size of the Convolution, in this case a 3x3 grid
# The activation function to use -- in this case we'll use relu, which you might recall is the equivalent of returning x when x>0, 
# else returning 0
# In the first layer, the shape of the input data.

# You'll follow the Convolution with a MaxPooling layer which is then designed to compress the image, 
# while maintaining the content of the features that were highlighted by the convlution. 
# By specifying (2,2) for the MaxPooling, the effect is to quarter the size of the image. 
# Without going into too much detail here, the idea is that it creates a 2x2 array of pixels, 
# and picks the biggest one, thus turning 4 pixels into 1. It repeats this across the image, 
# and in so doing halves the number of horizontal, and halves the number of vertical pixels, effectively reducing the image by 25%.

model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(28, 28, 1)),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2,2),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(10, activation='softmax')
])

# Now compile the model, call the fit method to do the training, and evaluate the loss and accuracy from the test set.
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# You can call model.summary() to see the size and shape of the network, 
# and you'll notice that after every MaxPooling layer, the image size is reduced in this way.
model.summary()
model.fit(training_images, training_labels, epochs=5)
test_loss = model.evaluate(test_images, test_labels)

