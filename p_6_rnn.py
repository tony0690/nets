import numpy
import pandas
import matplotlib.pyplot as plt
import sklearn
from tensorflow.keras.datasets import imdb
from tensorflow.keras import  layers,models
from tensorflow.keras.utils import to_categorical 
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.preprocessing import sequence

max_features = 10000
max_len = 500
batch_size = 32

(input_train, y_train), (input_test, y_test) = imdb.load_data(num_words=max_features)

input_train = sequence.pad_sequences(input_train, maxlen=max_len)
input_test = sequence.pad_sequences(input_test, maxlen=max_len)
  

model = models.Sequential()
model.add(Embedding(max_features, 32))  
model.add(layers.SimpleRNN(32))
model.add(Dense(1, activation='sigmoid')) 
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(input_train, y_train,
          epochs=10,
          batch_size=batch_size,
          validation_split=0.2)


score, acc = model.evaluate(input_test, y_test, batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)
