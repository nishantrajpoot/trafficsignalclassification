#!/usr/bin/env python
import os
import csv
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, MaxPool2D, Activation, Flatten
from skimage import io, color, exposure, transform
from sklearn.model_selection import train_test_split

# path
training = 'D:/Nishant/iv/task 1/GTSRB/Training/'


# function for reading the images
# arguments: path to the traffic sign data, for example './GTSRB/Training'
# returns: list of images, list of corresponding labels 
def readTrafficSigns(rootpath):
    '''Reads traffic sign data for German Traffic Sign Recognition Benchmark.

    Arguments: path to the traffic sign data, for example './GTSRB/Training'
    Returns:   list of images, list of corresponding labels'''
    images = []  # images
    labels = []  # corresponding labels
    # loop over all 43 classes
    for c in range(43):
        prefix = rootpath + '/' + format(c, '05d') + '/'  # subdirectory for class
        gtFile = open(prefix + 'GT-' + format(c, '05d') + '.csv')  # annotations file
        gtReader = csv.reader(gtFile, delimiter=';')  # csv parser for annotations file
        next(gtReader)  # skip header
        # loop over all images in current annotations file
        for row in gtReader:
            images.append(plt.imread(prefix + row[0]))  # the 1th column is the filename
            labels.append(row[7])  # the 8th column is the label
        gtFile.close()
    return images, labels


images, labels = readTrafficSigns(training)

# type(images[0])

avg_image_u = np.mean([x.shape[0] for x in images])
avg_image_v = np.mean([x.shape[1] for x in images])

# print(avg_image_u)
# print(avg_image_v)

# Preprocessing
class_size = 43
img_size = 48


def preprocess_img(img):
    # rescale to standard size
    img = transform.resize(img, (img_size, img_size))
    return img


X = [preprocess_img(x) for x in images]
labels = [int(x) for x in labels]

X = np.array(X)
Y = np.eye(43, dtype='uint8')[labels]


# Plot a histogram with the count of each traffic signs in the data set
plt.hist(labels, bins=43, edgecolor='black', width=0.5)
plt.title('Count of unique traffic signs')
plt.xlabel('Traffic Sign')
plt.ylabel('Number of samples')
plt.savefig('distribution of data.png')
plt.close()

# NOTE: Here the output shows mismatch the mismatch of data in class size.


# Defining the model
def cnn_model():
    model = Sequential()

    model.add(Conv2D(32, (3, 3), padding='same', input_shape=(img_size, img_size, 3), activation='relu'))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    # model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    # model.add(Conv2D(128, (3, 3), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(256, activation='relu', kernel_regularizer = tf.keras.regularizers.l2( l=0.01)))
    model.add(Dropout(0.5))
    model.add(Dense(class_size, activation='softmax'))
    return model


model = cnn_model()
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['acc', f1_m, precision_m, recall_m])

epochs = 25
callbacks = [
    keras.callbacks.ModelCheckpoint("model.h5", save_best_only=True)
]

X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2)

history = model.fit(X_train, Y_train, validation_data=(X_val,Y_val), batch_size=32, epochs=epochs, shuffle=True, callbacks=callbacks)

# model = keras.models.load_model('model.h5', custom_objects={"f1_m":f1_m, "precision_m":precision_m, "recall_m":recall_m})

# Plot model accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig('model_accuracy')

plt.close()
# plot model History
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('model_loss')

# prediction on test dataset
test = pd.read_csv('D:/Nishant/iv/task 1/test_images/GTSRB/Final_Test/Images/GT-final_test.csv', sep=';')

X_test = []
y_test = []
i = 0
for file_name, class_id in zip(list(test['Filename']), list(test['ClassId'])):
    img_path = os.path.join('test_images/GTSRB/Final_Test/Images', file_name)
    X_test.append(preprocess_img(io.imread(img_path)))
    y_test.append(class_id)

X_test = np.array(X_test)
y_test = np.array(y_test)

y_pred = model.predict_classes(X_test)
acc = np.sum(y_pred == y_test) / np.size(y_pred)
print("Test accuracy = {}".format(acc))

# To improve the accuracy, following points should be done:
# 1. correct the mismatch in the number of data for each class by adding augmented data.

