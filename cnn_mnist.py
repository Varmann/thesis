#https://www.geeksforgeeks.org/applying-convolutional-neural-network-on-mnist-dataset/

#Applying Convolutional Neural Network on mnist dataset

# CNN  learns through filters
#%%
import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Dense, Input
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten
from keras import backend as k

#%%
# Test data: Used for testing the model that how our model has been trained. 
# Train data: Used to train our model.

(x_train, y_train), (x_test, y_test) = mnist.load_data()

#%%
# Checking data-format: 

img_rows, img_cols=28, 28
 
if k.image_data_format() == 'channels_first':
   x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
   x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
   inpx = (1, img_rows, img_cols)
 
else:
   x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
   x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
   inpx = (img_rows, img_cols, 1)
 
 # In CNN, we can normalize data before hands such that large terms of the calculations can be reduced to smaller terms.
 #  Like, we can normalize the x_train and x_test data by dividing it by 255.
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

#%%
# Since the output of the model can comprise any of the digits between 0 to 9. so, we need 10 classes in output.

y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)

#%%
# CNN Model

inpx = Input(shape=inpx)
# layer1 is the Conv2d layer which convolves the image using 32 filters each of size (3*3). 
layer1 = Conv2D(32, kernel_size=(3, 3), activation='relu')(inpx)
# layer2 is again a Conv2D layer which is also used to convolve the image and is using 64 filters each of size (3*3). 
layer2 = Conv2D(64, kernel_size=(3, 3), activation='relu')(layer1)
# layer3 is the MaxPooling2D layer which picks the max value out of a matrix of size (3*3). 
layer3 = MaxPooling2D(pool_size=(3, 3))(layer2)
# layer4 is showing Dropout at a rate of 0.5. 
layer4 = Dropout(0.5)(layer3)
# layer5 is flattening the output obtained from layer4 and this flattens output is passed to layer6. 
layer5 = Flatten()(layer4)
# layer6 is a hidden layer of a neural network containing 250 neurons. 
layer6 = Dense(250, activation='sigmoid')(layer5)
# layer7 is the output layer having 10 neurons for 10 classes of output that is using the softmax function.
layer7 = Dense(10, activation='softmax')(layer6)


#%%
# Calling compile and fit function
model = Model([inpx], layer7)
model.compile(optimizer=keras.optimizers.Adadelta(),
              loss=keras.losses.categorical_crossentropy,
              metrics=['accuracy'])
 
#model_log = model.fit(x_train, y_train, epochs=1, batch_size=500)
model_log = model.fit(x_train, y_train, 
                    epochs=2,
                     batch_size=500)
          
# %%
score = model.evaluate(x_test, y_test, verbose=0)
print('loss=', score[0])
print('accuracy=', score[1])
# print((model_log.history.keys()))
#%%
import matplotlib.pyplot as plt
import os
# plotting the metrics
fig = plt.figure()
plt.subplot(2,1,1)
plt.plot(model_log.history['accuracy'])
plt.plot(model_log.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')
plt.subplot(2,1,2)
plt.plot(model_log.history['loss'])
plt.plot(model_log.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.tight_layout()
fig
#%%
# list all data in history
print(model_log.history.keys())
# summarize history for accuracy
plt.plot(model_log.history['accuracy'])
plt.plot(model_log.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(model_log.history['loss'])
plt.plot(model_log.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()