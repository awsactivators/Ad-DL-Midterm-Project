import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import fashion_mnist


def load_data():
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    return X_train, y_train, X_test, y_test


def create_model():
    model = keras.Sequential(
        [
            layers.Dense(128, activation="relu", input_shape=(784,)),
            layers.Dropout(0.3),
            layers.BatchNormalization(),
            layers.Dense(24, activation="relu"),
            layers.Dropout(0.3),
            layers.BatchNormalization(),
            layers.Dense(24, activation="relu"),
            layers.Dropout(0.3),
            layers.BatchNormalization(),
            layers.Dense(10, activation="softmax"),
        ]
    )

    return model


def compile_and_train(
    model,
    X_train,
    y_train,
    X_test,
    y_test,
    optimizer,
    batch_size=100,
    epochs=30,
    validation_split=0.2,
):
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=optimizer,
        metrics=["accuracy"],
    )
    history = model.fit(
        X_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=validation_split,
    )

    # Evaluate the model on the test set
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"Test loss: {test_loss}, Test accuracy: {test_acc}")

    return history


def evaluate_model(model, X_test, y_test):
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"Test loss: {test_loss}, Test accuracy: {test_acc}")

    return test_loss, test_acc


def reshape_data(X_train, X_test):
    x_train, x_test = X_train / 255.0, X_test / 255.0
    x_train_flattened = x_train.reshape(x_train.shape[0], -1)
    x_test_flattened = x_test.reshape(x_test.shape[0], -1)
    return x_train_flattened, x_test_flattened
