import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D, LeakyReLU
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import sys
from time import time
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score

num_classes = 10
input_shape = (32, 32, 3)

history_list = []; training_time_list = []
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
n = 5000
x_train = x_train[1:n]; y_train=y_train[1:n]

x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
print("orig x_train shape:", x_train.shape)

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

history_list = []
use_saved_model = False
if use_saved_model:
    model = keras.models.load_model("cifar.model")
else:
    model = keras.Sequential()
    model.add(Conv2D(16, (3,3), padding='same', input_shape=x_train.shape[1:],activation='relu'))
    model.add(Conv2D(16, (3,3), strides=(2,2), padding='same', activation='relu'))
    model.add(Conv2D(32, (3,3), padding='same', activation='relu'))
    model.add(Conv2D(32, (3,3), strides=(2,2), padding='same', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax',kernel_regularizer=regularizers.l1(0.0001)))
    model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=["accuracy"])
    model.summary()
    batch_size = 128
    epochs = 20
    start_time = time()
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)
    training_time = (time() - start_time)
    history_list.append(history)
    training_time_list.append(training_time)
    model.save('cifar.model')
    
use_saved_model = False
if use_saved_model:
    model = keras.models.load_model("cifar.model")
else:
    model = keras.Sequential()
    model.add(Conv2D(16, (3,3), padding='same', input_shape=x_train.shape[1:],activation='relu'))
    model.add(Conv2D(16, (3,3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2), padding='same'))
    model.add(Conv2D(32, (3,3), padding='same', activation='relu'))
    model.add(Conv2D(32, (3,3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2), padding='same'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax',kernel_regularizer=regularizers.l1(0.0001)))
    model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=["accuracy"])
    model.summary()
    batch_size = 128
    epochs = 20
    start_time = time()
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)
    training_time = (time() - start_time)
    history_list.append(history)
    training_time_list.append(training_time)
    model.save('cifar.model')

dummy = DummyClassifier(strategy='most_frequent')
y_train_flatten = np.argmax(y_train, axis=-1)
dummy.fit(x_train, y_train_flatten)
pred = dummy.predict(x_test)
y_test_flatten = np.argmax(y_test, axis=-1)
acc_baseline = accuracy_score(pred, y_test_flatten)
acc_baseline = [acc_baseline] * epochs

fig = plt.figure(figsize=(16, 8))
ax = fig.add_subplot(1, 2, 1)
alpha_list = [0.25, 1.0]
for history, alpha in zip(history_list, alpha_list):
    ax.plot(history.history['val_loss'])
ax.set_title('Model Loss', fontsize=14)
ax.set_ylabel('Loss', fontsize=12)
ax.set_xlabel('Epochs', fontsize=12)
ax.legend(['Stride', 'MaxPooling', 'Baseline'], loc='upper right')
ax = fig.add_subplot(1, 2, 2)
alpha_list = [0.25, 1.0]
for history, alpha in zip(history_list, alpha_list):
    ax.plot(history.history['val_accuracy'])
ax.plot(acc_baseline)
ax.set_title('Model Accuracy', fontsize=14)
ax.set_ylabel('Accuracy', fontsize=12)
ax.set_xlabel('Epochs', fontsize=12)
ax.legend(['Stride', 'MaxPooling', 'Baseline'], loc='lower right')
plt.tight_layout()
plt.savefig('evaluation_of_maxpooling')
plt.show()