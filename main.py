import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt

# This code was created when following the MIT Course on deeplearning
# https://ocw.mit.edu/courses/15-773-hands-on-deep-learning-spring-2024/video_galleries/lecture-videos/
#
# colabs
# https://colab.research.google.com/drive/1S2tt-klwRH4czQcjsbLWISG9KPppUVEr?usp=sharing

def main():
    # URL = "http://storage.googleapis.com/download.tensorflow.org/data/heart.csv"
    URL = "heart.csv"
    proxies = { "http" : "http://127.0.0.1:8080" }
    # re = requests.get(url=URL, proxies=proxies)
    df = pd.read_csv(URL)
    print(df.target.value_counts(normalize=True, dropna=False))
    categoriical_variables = ['sex','cp','fbs','restecg','exang','ca','thal']
    numerics = ['age','trestbps','chol','thalach','oldpeak','slope']
    df = pd.get_dummies(df, columns=categoriical_variables)
    print(df.info)
    test_df = df.sample(frac=0.2, random_state=42)
    train_df = df.drop(test_df.index)
    means = train_df[numerics].mean()
    sd = train_df[numerics].std()
    train_df[numerics] = (train_df[numerics] - means)/sd
    test_df[numerics] = (test_df[numerics] - means)/sd

    train = train_df.to_numpy()
    test = test_df.to_numpy()

    train_X = np.delete(train, 6, axis=1)
    train_X = np.asarray(train_X).astype(np.int_)
    test_X = np.delete(test, 6, axis=1)
    test_X = np.asarray(test_X).astype(np.int_)
    
    train_y = train[:, 6]
    train_y = np.asarray(train_y).astype(np.int_)
    # the line above I found on https://datascience.stackexchange.com/questions/82440/valueerror-failed-to-convert-a-numpy-array-to-a-tensor-unsupported-object-type
    # without this line, I get a invalid dtype: object error

    test_y = test[:, 6]
    test_y = np.asarray(test_y).astype(np.int_)

    num_columns = (train_X.shape[1],)

    input = keras.Input(shape=(29,))
    h = keras.layers.Dense(16, activation="relu", name="Hidden")(input)
    output = keras.layers.Dense(1, activation="sigmoid", name="Output")(h)
    model = keras.Model(input, output)
    model.summary()



    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    history = model.fit(train_X, train_y, epochs=300, batch_size=32, verbose=1, validation_split=0.2)
    print(train_X)

    history_dict = history.history
    history_dict.keys()

    los_values = history_dict["loss"]
    val_loss_values = history_dict["val_loss"]
    epochs = range(1, len(los_values) +1)
    plt.plot(epochs, los_values, "bo", label="Training loss")
    plt.plot(epochs, val_loss_values, "b", label="Validation loss")
    plt.title("Training and validation loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
    # error with display where solved with export QT_QPA_PLATFORM=''


    model.evaluate(test_X, test_y)
    
    

if __name__ == "__main__":
    main()


