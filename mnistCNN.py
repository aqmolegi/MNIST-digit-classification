import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Activation, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data() # load mnist dataset

num_labels = len(np.unique(y_train)) # get the unique number of labels

y_train = to_categorical(y_train) # convert the train labels to one-hot vector 
y_test = to_categorical(y_test) # convert the test labels to one-hot vector 
  
image_size = x_train.shape[1] # get the image dimensions
input_size = (image_size, image_size, 1)

x_train = np.reshape(x_train,[-1, image_size, image_size, 1]) # resize the train data
x_test = np.reshape(x_test,[-1, image_size, image_size, 1]) # resize the test data
x_train = x_train.astype('float32') / 255 # normalize the train data
x_test = x_test.astype('float32') / 255 # normalize the test data

# network hyperparameters
batch_size = 200
kernel_size = 3
pool_size = 2
filters = 64
dropout = 0.2

# CNN model layers
model = Sequential()
model.add(Conv2D(filters=filters,
                 kernel_size=kernel_size,
                 activation='relu',
                 input_shape=input_size))
model.add(MaxPooling2D(pool_size))
model.add(Conv2D(filters=filters,
                 kernel_size=kernel_size,
                 activation='relu'))
model.add(MaxPooling2D(pool_size))
model.add(Conv2D(filters=filters,
                 kernel_size=kernel_size,
                 activation='relu'))
model.add(Flatten())
model.add(Dropout(dropout))
model.add(Dense(num_labels))
model.add(Activation('softmax'))
model.summary()
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=20, batch_size=batch_size) #  train the network

loss, acc = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=0) # validate the model
print("\nTesting accuracy: %.1f%%" % (100.0 * acc))