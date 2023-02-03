#%% Imports
import os
import numpy as np
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils import to_categorical
from matplotlib import pyplot as plt


#%%          # 28x28 Bilder mit handgeschriebenen Ziffern (0-9)
mnist = keras.datasets.mnist 

#%%       # Splitten des Datensatzes in Training_ und Testdaten
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# %%  Trainingsdata anzeigen
#n = 10
#print(y_train[n]),  
#plt.imshow(x_train[n])  

# %% Normalisierung der Daten zwischen 0 und 1
x_train = keras.utils.normalize(x_train, axis=1)
x_test = keras.utils.normalize(x_test, axis=1)
#print (x_train)

# %% Erstellung und Training des Künstlichen neuronalen Netzes. 
model = Sequential()
model.add(tf.keras.layers.Flatten())
model.add (Dense(128, activation = 'relu'))
model.add (Dense(128, activation = 'relu'))
#Gibt Wahrscheinlichkeiten einzelnen Ziffern aus
model.add (Dense(10, activation = 'softmax')) 

# %%  Definition der Parameter für Training
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])


# %% Training des Models , mit allen Trainingsdaten ,  784 Pixel (28x28)
model.fit(x_train.reshape(-1,784),y_train, epochs= 6 ) 

# %%  Erkennung    # -1 Alle Testdaten 
var_loss, var_accuracy  = model.evaluate(x_test.reshape(-1,784), y_test)
print(var_loss, var_accuracy)

# %% Was befindet sich laut unserem Neuronalen Netz auf Bild 21 des Testdatensaty
predictions = model.predict(x_test.reshape(-1,784))
import random 
import time
n =0 
for i in range(10):
   
    #input_num = input("Nummer eingeben  ")
    #i= int(input_num)
    
    i = random.randint(0,1000)
    print(predictions[i])
    pred_i = predictions[i]
    argmax_i = np.argmax(pred_i)
    print(argmax_i)
    fig, axs = plt.subplots(1)
   
    axs.set_title("Prediction : " + str(round(100*pred_i[argmax_i],3))+ "%  ist  Zahl  " + str(argmax_i))
    plt.imshow(x_test[i])
    plt.show()
    #time.sleep(0.5)
    plt.close()

# %%
