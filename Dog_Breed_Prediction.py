# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 17:30:30 2021

@author: Ahmed Fayed
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split

from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D
from keras.optimizers import Adam



# Reading and exploring labels
labels = pd.read_csv('labels.csv')


breeds = labels["breed"]
breed_counts = breeds.value_counts()

classes_names = breeds.unique()

CLASS_NAMES = ['scottish_deerhound','maltese_dog','bernese_mountain_dog']
labels = labels[(labels['breed'].isin(CLASS_NAMES))]
labels = labels.reset_index()


X_data = np.zeros((len(labels), 224, 224, 3), dtype='float32')

Y_data = label_binarize(labels['breed'], classes = classes_names)

for i in range(len(labels)):
    img = image.load_img('train/%s.jpg' % labels['id'][i], target_size = (224, 224))
    img = image.img_to_array(img)
    
    x = np.expand_dims(img.copy(), axis=0)
    X_data[i] = x / 255.0
    

model = Sequential()

model.add(Conv2D(filters = 128, kernel_size=(7, 7), activation='relu', input_shape = (224, 224, 3)))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Conv2D(filters = 64, kernel_size=(5, 5), activation='relu', kernel_regularizer='l2'))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Conv2D(filters = 32, kernel_size=(3, 3), kernel_regularizer='l2'))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Conv2D(filters = 16, kernel_size=(3, 3), activation='relu', kernel_regularizer='l2',))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(1024, activation='relu', kernel_regularizer='l2'))
model.add(Dense(1024, activation='relu', kernel_regularizer='l2'))
model.add(Dense(len(classes_names), activation='softmax'))

model.compile(optimizer = Adam(0.0001), loss = 'categorical_crossentropy', metrics=['accuracy'])

model.summary()

X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size=0.20, shuffle = True)

# splitting training data into training and validation datasets
# X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2)


epochs = 100
batch_size = 128

history = model.fit(X_train, 
                    Y_train, 
                    batch_size=batch_size, 
                    epochs=epochs, #validation_data=(X_val, Y_val)
                    )



plt.figure(figsize=(12, 5))
plt.plot(history.history['accuracy'], color='b')
# plt.plot(history.history['val_accuracy'], color='y')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel("Epochs")
plt.legend(['train'])
plt.show()

Y_pred = model.predict(X_test)
score = model.evaluate(X_test, Y_test)

print('Test set Accuracy: ', round(score[1] * 100, 2), '%')



# plotting image for the comparison

plt.imshow(X_test[1, :, :, :])
plt.show()

# Finding max value from predition list and comaparing original value vs predicted
print('Original : ',labels['breed'][np.argmax(Y_test[1])])
print('Predicted: ', labels['breed'][np.argmax(Y_pred[1])])



















