import tensorflow as tf
import matplotlib.pyplot as plt


class FMM:
    @staticmethod
    def load_data():
        fashion_mnist = tf.keras.datasets.fashion_mnist
        (X_train, y_train), (
            X_test,
            y_test,
        ) = fashion_mnist.load_data()
        return X_train, y_train, X_test, y_test

    @staticmethod
    def create_model():
        return tf.keras.Sequential(
            [
                tf.keras.layers.Flatten(input_shape=(28, 28)),
                tf.keras.layers.Dense(256, activation="relu"),
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.Dense(10, activation="softmax"),
            ]
        )


    @staticmethod
    def reshape_data(X_train, X_test):
        return X_train / 255.0, X_test / 255.0

    @staticmethod
    def compile_and_train(
        model,
        X_train,
        y_train,
        X_test,
        y_test,
        optimizer,
        batch_size=100,
        epochs=60,
    ):
        model.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            optimizer=optimizer,
            metrics=["accuracy"],
        )
        history = model.fit(
            X_train,
            y_train,
            batch_size=batch_size,
            epochs=epochs,
            shuffle=True,
            validation_data=(X_test, y_test),
        )

        train_accuracy = history.history["accuracy"][-1]
        val_accuracy = history.history["val_accuracy"][-1]

        return history, train_accuracy, val_accuracy

    @staticmethod
    def evaluate(
        model,
        X_test,
        y_test,
    ):
        loss, accuracy = model.evaluate(X_test, y_test, verbose=2)
        return loss, accuracy

    @staticmethod
    def plot_history(history, optimizer):
        plt.plot(history.history["loss"], label="Training loss")
        plt.plot(history.history["val_loss"], label="Validation loss")
        plt.title(f"Training and Validation Loss ({optimizer.__class__.__name__})")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()
