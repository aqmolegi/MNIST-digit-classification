import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, SimpleRNN, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data() # load mnist dataset

num_labels = len(np.unique(y_train)) # get the unique number of labels

y_train = to_categorical(y_train) # convert the train labels to one-hot vector 
y_test = to_categorical(y_test) # convert the test labels to one-hot vector 

image_size = x_train.shape[1] # get the image dimensions
input_size = (image_size, image_size)

x_train = np.reshape(x_train,[-1, image_size, image_size]) # resize the train data
x_test = np.reshape(x_test,[-1, image_size, image_size]) # resize the test data
x_train = x_train.astype('float32') / 255 # normalize the train data
x_test = x_test.astype('float32') / 255 # normalize the test data

# network hyperparameters
batch_size = 200
units = 256
dropout = 0.2

# RNN model layers
model = Sequential()
model.add(SimpleRNN(units=units,
                    dropout=dropout,
                    activation='relu',
                    input_shape=input_size))
model.add(Dense(num_labels))
model.add(Activation('softmax'))
model.summary()
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=20, batch_size=batch_size) #  train the network

loss, acc = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=0) # validate the model
print("\nTesting accuracy: %.1f%%" % (100.0 * acc))