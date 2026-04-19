import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

keras.utils.set_random_seed(42)

(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

labels = ["T-Shirt.top",
          "Trouser",
          "Pullover",
          "Dress",
          "Coat",
          "Sandal",
          "Shirt",
          "Sneaker",
          "Bag",
          "Ankle Boot"]

fig, ax = plt.subplots(5, 5, figsize=(30, 10))
for i in range(25):
    axc = ax[i//5, i%5]
    axc.imshow(x_train[i], cmap="gray")
    axc.set_title(f"{labels[y_train[i]]}")
    axc.set_xticks([])
    axc.set_yticks([])

# plt.show()


def plot_loss_curves(history):
    plt.clf()
    history_dict = history.history
    loss_values = history_dict["loss"]
    val_loss_values = history['val_loss']
    epochs = range(1, len(loss_values) +1)
    plt.plot(epochs, loss_values, "bo", label="Training loss")
    plt.plot(epochs, val_loss_values, "b", label="Validation Loss")
    plt.title("Training and validation loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

def plot_acc_curves(history):
    plt.clf()
    history_dict = history.history
    acc = history_dict["accuracy"]
    val_acc = history_dict["val_accuracy"]
    epochs = range(1, len(acc) +1)
    plt.plot(epochs, acc, "bo", label="Training ACC")
    plt.plot(epochs, val_acc, "b", label="Validation ACC")
    plt.title("Training and validation accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

x_train = x_train/255.0
x_test = x_test/255.0

x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
x_train.shape

input = keras.Input(shape=x_train.shape[1:])
x = keras.layers.Conv2D(32, kernel_size=(2,2), activation="relu", name="Conv_1")(input)
x = keras.layers.MaxPool2D()(x)

x = keras.layers.Conv2D(32, kernel_size=(2,2), activation="relu", name="Conv_2")(input)
x = keras.layers.MaxPool2D()(x)

x = keras.layers.Flatten()(x)

x = keras.layers.Dense(256, activation="relu")(x)

output = keras.layers.Dense(10, activation="softmax")(x)
model = keras.Model(input, output)

model.summary()

model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

history = model.fit(x_train, y_train, batch_size=64, epochs=10, validation_split=0.2)

plot_acc_curves(history)

score = model.evaluate(x_test, y_test)
print("Test Accuracy :", score[1])