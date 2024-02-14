import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical

def load_data():
    fashion_train = pd.read_csv("fashion_mnist/fashion-mnist_train.csv")
    fashion_test = pd.read_csv("fashion_mnist/fashion-mnist_test.csv")

    X_train = fashion_train.drop('label', axis=1)
    y_train = to_categorical(fashion_train['label'], num_classes=10)
    X_test = fashion_test.drop('label', axis=1)
    y_test = to_categorical(fashion_test['label'], num_classes=10)

    return X_train, y_train, X_test, y_test

def create_model():
    model = keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(784,)),
        layers.Dropout(0.3),
        layers.BatchNormalization(),
        layers.Dense(24, activation='relu'),
        layers.Dropout(0.3),
        layers.BatchNormalization(),
        layers.Dense(24, activation='relu'),
        layers.Dropout(0.3),
        layers.BatchNormalization(),
        layers.Dense(10, activation='softmax'),
    ])
    return model

def compile_and_train(model, X_train, y_train, optimizer, batch_size=100, epochs=30):
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=['accuracy'])
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs)

def evaluate_model(model, X_test, y_test):
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print('Test accuracy:', round(test_acc, 4))
    return test_loss, test_acc
