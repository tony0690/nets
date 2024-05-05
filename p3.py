import numpy
import pandas
import matplotlib.pyplot as plt
import sklearn
from tensorflow.keras.datasets import mnist
from tensorflow.keras import  layers,models
from tensorflow.keras.utils import to_categorical 

(train_img,train_labels),(test_img,test_labels) = mnist.load_data()
train_img = train_img.reshape((60000,28,28,1)).astype('float32')/255
test_img = test_img.reshape((10000,28,28,1)).astype('float32')/255
from tensorflow.keras import models, layers

class NeuralNet:
    def __init__(self, input_shape, num_classes):
        self.model = self.build_model(input_shape, num_classes) 

    def build_model(self, input_shape, num_classes):
        model = models.Sequential()
        model.add(layers.Flatten(input_shape = input_shape))
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(num_classes, activation='softmax'))
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def train(self, train_img, train_labels, epochs=5, batch_size=64, validation_split=0.1):
        history = self.model.fit(train_img, train_labels, epochs=epochs, batch_size=batch_size, validation_split=validation_split)
        return history

    def eval(self, test_img, test_labels):
        return self.model.evaluate(test_img, test_labels)

    def pred(self, img):
        return self.model.predict(img)



input_shape = (28, 28,1)  # Corrected input_shape format
num_classes = 10
nn = NeuralNet(input_shape, num_classes)
histor = nn.train(train_img, train_labels)
test_loss, test_acc = nn.eval(test_img, test_labels)
print(f'Test accuracy: {test_acc}')
predictions = nn.pred(test_images[:5])


for i in range(5):
  plt.imshow(test_images[i])
  plt.title(f'Predicted: {tf.argmax(predictions[i])}')

