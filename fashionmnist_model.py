import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple


class FMM:
    @staticmethod
    def load_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Function to load the Fashion MNIST dataset.

        Returns
        - tuple: A tuple containing four numpy arrays::
            - X_train (numpy array): Training images.
            - y_train (numpy array): Training labels.
            - X_test (numpy array): Test images.
            - y_test (numpy array): Test labels.
        """

        fashion_mnist = tf.keras.datasets.fashion_mnist
        (X_train, y_train), (
            X_test,
            y_test,
        ) = fashion_mnist.load_data()
        return X_train, y_train, X_test, y_test

    @staticmethod
    def get_model():
        model = FMM.create_model_v1()
        print(f"Training with model {model.name} ...")
        return model

    @staticmethod
    def create_model_v1() -> tf.keras.Sequential:
        """
        Function to create a simple neural network model (version 1) using TensorFlow Keras.

        Returns:
        - A TensorFlow Keras Sequential model object representing the neural network.

        This model consists of the following layers:
        - Flatten: Flattens the input images from 2D arrays (28x28 pixels) to 1D arrays (784 pixels).
        - Dense: Fully connected layer with 256 neurons, using ReLU activation function.
        - Dense: Fully connected layer with 64 neurons, using ReLU activation function.
        - Dense: Output layer with 10 neurons (one for each class in the dataset), using softmax activation function.

        The model is named "model_v1".
        """

        return tf.keras.Sequential(
            [
                tf.keras.layers.Flatten(input_shape=(28, 28)),
                tf.keras.layers.Dense(256, activation="relu"),
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.Dense(10, activation="softmax"),
            ],
            name="model_v1",
        )

    @staticmethod
    def create_model_v2() -> tf.keras.Sequential:
        return tf.keras.Sequential(
            [
                tf.keras.layers.Flatten(input_shape=(28, 28)),
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(32, activation="relu"),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(10, activation="softmax"),
            ],
            name="model_v2",
        )

    @staticmethod
    def create_model_v3() -> tf.keras.Sequential:
        return tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(
                    32,
                    (3, 3),
                    padding="same",
                    input_shape=(28, 28, 1),
                    activation="relu",
                ),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.25),
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.25),
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.25),
                tf.keras.layers.Dense(10, activation="softmax"),
            ],
            name="model_v3",
        )

    @staticmethod
    def reshape_data(
        X_train: np.ndarray, X_test: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Reshape the input data by scaling it to the range [0, 1].

        Parameters:
        - X_train (numpy.ndarray): The training data.
        - X_test (numpy.ndarray): The testing data.

        Returns:
        - tuple: A tuple containing two numpy arrays:
            - X_train_scaled: The scaled training data.
            - X_test_scaled: The scaled testing data.
        """

        return X_train / 255.0, X_test / 255.0

    @staticmethod
    def compile_and_train(
        model: tf.keras.Model,
        X_train: np.ndarray,
        y_train: np.ndarray,
        optimizer: tf.keras.optimizers.Optimizer,
        batch_size: int = 64,
        epochs: int = 30,
        validation_split: float = 0.2,
    ):
        """
        Compiles the given model and trains it on the provided training data.

        Args:
        - model (tf.keras.Model): The neural network model to be compiled and trained.
        - X_train (numpy.ndarray): The input training data.
        - y_train (numpy.ndarray): The target training data.
        - optimizer (tf.keras.optimizers.Optimizer): The optimizer to use during training.
        - batch_size (int): The batch size for training. Default is 64.
        - epochs (int): The number of epochs for training. Default is 30.
        - validation_split (float): The fraction of training data to use as validation data. Default is 0.2.

        Returns:
        - history (tf.keras.callbacks.History): A History object containing training metrics.

        Note:
        - If an optimizer is provided, the model is compiled using the specified optimizer.
        - If no optimizer is provided, the model is compiled without any optimizer.
        - Uses SparseCategoricalCrossentropy loss and accuracy metrics for compilation.
        - The model is trained using fit() method with the provided parameters.
        """

        if optimizer:
            model.compile(
                loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                optimizer=optimizer,
                metrics=["accuracy"],
            )
        else:
            model.compile(
                loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                metrics=["accuracy"],
            )

        history = model.fit(
            X_train,
            y_train,
            batch_size=batch_size,
            epochs=epochs,
            shuffle=True,
            validation_split=validation_split,
        )

        return history

    @staticmethod
    def evaluate(
        model: tf.keras.Model, X_test: np.ndarray, y_test: np.ndarray, history
    ) -> Tuple[float, float, float, float]:
        """
        Evaluate the trained model on the test data and print out the results.

        Parameters:
        - model (keras.Model): The trained neural network model.
        - X_test (numpy.ndarray): Input features for testing.
        - y_test (numpy.ndarray): True labels for testing.
        - history (keras.History): History object containing training metrics.

        Returns:
        - loss (float): The loss value on the test set.
        - accuracy (float): The accuracy on the test set.
        - train_accuracy (float): The final accuracy achieved during training.
        - val_accuracy (float): The final validation accuracy achieved during training.
        """

        # Get the final training and validation accuracy from the training history
        train_accuracy = history.history["accuracy"][-1]
        val_accuracy = history.history["val_accuracy"][-1]

        # Evaluate the model on the test set and get the loss and accuracy
        loss, accuracy = model.evaluate(X_test, y_test, verbose=2)

        # Print out the evaluation results
        print(f"\nTraining accuracy : {train_accuracy}")
        print(f"Validation accuracy : {val_accuracy}")
        print(f"Loss : {loss}")
        print(f"Accuracy : {accuracy}\n")

        # Return the evaluation results
        return loss, accuracy, train_accuracy, val_accuracy

    @staticmethod
    def plot_history(history):
        """
        Function to plot training and validation accuracy as well as training and validation loss.

        Parameters:
        - history: A Keras History object containing training history.

        Returns:
        - None

        Plots:
        - Training and validation accuracy curves.
        - Training and validation loss curves.
        """

        # Plot training & validation accuracy values
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(history.history["accuracy"], label="Training Accuracy")
        plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
        plt.title(f"Model Accuracy")
        plt.ylabel("Accuracy")
        plt.xlabel("Epoch")
        plt.legend()

        # Plot training & validation loss values
        plt.subplot(1, 2, 2)
        plt.plot(history.history["loss"], label="Training Loss")
        plt.plot(history.history["val_loss"], label="Validation Loss")
        plt.title(f"Model Loss")
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.legend()

        plt.show()

    @staticmethod
    def plot_analysis_results(analysis, x_axis, y_axis):
        all_results = analysis.trial_dataframes

        plt.figure(figsize=(12, 6))
        for trial in all_results.values():
            plt.plot(
                trial[f"{x_axis}"],
                trial[f"{y_axis}"],
                marker="o",
                label=f"Trial {trial['trial_id'][0]}",
            )

        plt.title(f"{y_axis} vs {x_axis}")
        plt.xlabel(f"{x_axis}")
        plt.ylabel(f"{y_axis}")
        plt.legend()
        plt.grid(True)
        plt.show()
