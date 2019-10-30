from keras.datasets import imdb
from keras import models
from tensorflow.keras import layers
from keras import layers

import numpy as np

def vectorize_sequences(sequences, dimension=10000):
    # Create an all-zero matrix of shape (len(sequences), dimension)
    print("shape=%s" % sequences.shape)
    results = np.zeros((len(sequences), dimension), dtype='uint16')
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.  # set specific indices of results[i] to 1s
    return results

#load data
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

# vectorize train data
x_train = vectorize_sequences(train_data)

# vectorize test data
x_test = vectorize_sequences(test_data)

# vectorize train label
y_train = np.asarray(train_labels).astype('float32')

# vectorize test label
y_test = np.asarray(test_labels).astype('float32')

model = models.Sequential()

model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop',
             loss='binary_crossentropy',
             metrics=['accuracy'])

x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_val, y_val)
                   )
