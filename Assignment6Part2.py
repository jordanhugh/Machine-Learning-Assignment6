import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D, LeakyReLU
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.utils import shuffle
from sklearn.dummy import DummyClassifier
import matplotlib.pyplot as plt
import sys
from time import time

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

    dummy = DummyClassifier(strategy='most_frequent')
    y_train_flatten = np.argmax(y_train, axis=-1)
    dummy.fit(x_train, y_train_flatten)
    pred = dummy.predict(x_test)
    y_test_flatten = np.argmax(y_test, axis=-1)
    acc_baseline = accuracy_score(pred, y_test_flatten)
    acc_baseline = [acc_baseline] * epochs

    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_subplot(1, 2, 1)
    ax.plot(history.history['loss'])
    ax.plot(history.history['val_loss'])
    ax.set_title('Model Loss', fontsize=14)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_xlabel('Epochs', fontsize=12)
    ax.legend(['Training', 'Validation', 'Baseline'], loc='upper right')
    ax = fig.add_subplot(1, 2, 2)
    ax.plot(history.history['accuracy'])
    ax.plot(history.history['val_accuracy'])
    ax.plot(acc_baseline)
    ax.set_title('Model Accuracy', fontsize=14)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_xlabel('Epochs', fontsize=12)
    ax.legend(['Training', 'Validation', 'Baseline'], loc='lower right')
    plt.tight_layout()
    plt.savefig('evaluation_against_baseline')   
    plt.show()

        

size_list = [5000, 10000, 20000, 40000]
history_list = []; training_time_list = []
for size in size_list:
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    x_train = x_train[1:size]; y_train=y_train[1:size]

    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    print("orig x_train shape:", x_train.shape)

    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

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
        
        fig = plt.figure(figsize=(16, 8))
        ax = fig.add_subplot(1, 2, 1)
        ax.plot(history.history['loss'])
        ax.plot(history.history['val_loss'])
        ax.set_title('Model Loss', fontsize=14)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_xlabel('Epochs', fontsize=12)
        ax.legend(['Training', 'Validation'], loc='upper right')
        ax = fig.add_subplot(1, 2, 2)
        ax.plot(history.history['accuracy'])
        ax.plot(history.history['val_accuracy'])
        ax.set_title('Model Accuracy', fontsize=14)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_xlabel('Epochs', fontsize=12)
        ax.legend(['Training', 'Validation'], loc='lower right')
        plt.tight_layout()
        plt.savefig('training_size_' + str(size))   
        plt.show()
         
fig = plt.figure(figsize=(16, 8))
ax = fig.add_subplot(1, 2, 1)
accuracy_list = [];
for history in history_list:
    accuracy_list.append(history.history['accuracy'][-1])
ax.plot(size_list, accuracy_list)
ax.set_title('Model Accuracy', fontsize=14)
ax.set_ylabel('Accuracy', fontsize=12)
ax.set_xlabel('Size of Training Set', fontsize=12)
ax = fig.add_subplot(1, 2, 2)
ax.plot(size_list, training_time_list)
ax.set_title('Model Training Time', fontsize=14)
ax.set_ylabel('Time (s)', fontsize=12)
ax.set_xlabel('Size of Training Set', fontsize=12)
plt.tight_layout()
plt.savefig('evaluation_of_varying_size')   
plt.show()

        
        
penalty_list = [0, 0.1, 0.5, 1, 5, 10, 50, 100]
history_list = []; training_time_list = []
for penalty in penalty_list:
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    x_train = x_train[1:n]; y_train=y_train[1:n]

    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    print("orig x_train shape:", x_train.shape)

    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

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
        model.add(Dense(num_classes, activation='softmax',kernel_regularizer=regularizers.l1(penalty)))
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
        
loss_list = []; val_loss_list = []
acc_list = []; val_acc_list = []
for history in history_list:
    loss_list.append(history.history['loss'][-1])
    val_loss_list.append(history.history['val_loss'][-1])
    acc_list.append(history.history['accuracy'][-1])
    val_acc_list.append(history.history['val_accuracy'][-1])
    
fig = plt.figure(figsize=(16, 8))
ax = fig.add_subplot(1, 2, 1)
ax.plot(penalty_list, loss_list)
ax.plot(penalty_list, val_loss_list)
ax.set_title('Model Loss', fontsize=14)
ax.set_ylabel('Loss', fontsize=12)
ax.set_xlabel('Alpha', fontsize=12)
ax.legend(['Training', 'Validation'], loc='lower right')
ax = fig.add_subplot(1, 2, 2)
ax.plot(penalty_list, acc_list)
ax.plot(penalty_list, val_acc_list)
ax.set_title('Model Accuracy', fontsize=14)
ax.set_ylabel('Accuracy', fontsize=12)
ax.set_xlabel('Alpha', fontsize=12)
ax.legend(['Training', 'Validation'], loc='lower right')
plt.tight_layout()
plt.savefig('evaluation_of_varying_penalty')   
plt.show()