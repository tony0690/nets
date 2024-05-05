import numpy
import pandas
import matplotlib.pyplot as plt
import sklearn
from tensorflow.keras.datasets import mnist
from tensorflow.keras import  layers,models
from tensorflow.keras.utils import to_categorical 

 
  
(train_img, train_labels),(test_img,test_labels) = mnist.load_data()
train_img = train_img.reshape((60000,28,28,1)).astype('float32')/255
test_img = test_img.reshape((10000,28,28,1)).astype('float32')/255
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

model = models.Sequential()
model.add(layers.Conv2D(32,(3,3),activation = 'relu', input_shape = (28,28,1)))
model.add(layers.MaxPool2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation = 'relu'))
model.add(layers.MaxPool2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation = 'relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64,activation = 'relu'))
model.add(layers.Dense(10, activation='softmax'))
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy',metrics = ['accuracy'])

test_loss, test_acc = model.evaluate(test_img, test_labels)
print(f'Test accuracy: {test_acc}')


predictions = model.predict(test_images[:5])
for i in range(5):
    plt.imshow(test_images[i].reshape(28, 28), cmap='gray')
    plt.title(f'Predicted: {tf.argmax(predictions[i])}, Actual: {tf.argmax(test_labels[i])}')
    plt.show()

