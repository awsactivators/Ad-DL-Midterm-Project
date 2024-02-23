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
    def create_model_v1():
        return tf.keras.Sequential(
            [
                tf.keras.layers.Flatten(input_shape=(28, 28)),
                tf.keras.layers.Dense(256, activation="relu"),
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.Dense(10, activation="softmax"),
            ]
        )

    @staticmethod
    def create_model_v2():
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
            ]
        )

    @staticmethod
    def create_model_v3():
        return tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(32, (3,  3), padding='same', input_shape=(28,  28,  1), activation='relu'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.25),
                tf.keras.layers.MaxPooling2D(pool_size=(2,  2)),

                tf.keras.layers.Conv2D(64, (3,  3), padding='same', activation='relu'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.25),
                tf.keras.layers.MaxPooling2D(pool_size=(2,  2)),

                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.25),
                
                tf.keras.layers.Dense(10, activation='softmax'),
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
        optimizer,
        batch_size=64,
        epochs=30,
        validation_split=0.2,
    ):
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
            validation_split=validation_split
        )

        return history

    @staticmethod
    def evaluate(
        model,
        X_test,
        y_test,
        history
    ):
        train_accuracy = history.history["accuracy"][-1]
        val_accuracy = history.history["val_accuracy"][-1]
    
        loss, accuracy = model.evaluate(X_test, y_test, verbose=2)

        print(f"\nTraining accuracy : {train_accuracy}")
        print(f"Validation accuracy : {val_accuracy}")
        print(f"Loss : {loss}")
        print(f"Accuracy : {accuracy}\n")

        return loss, accuracy, train_accuracy, val_accuracy

    @staticmethod
    def plot_history(history):
       
        # Plot training & validation accuracy values
        plt.figure(figsize=(12,  6))
        plt.subplot(1,  2,  1)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title(f'Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend()

        # Plot training & validation loss values
        plt.subplot(1,  2,  2)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title(f'Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()

        plt.show()

    @staticmethod
    def plot_analysis_results(analysis, x_axis, y_axis):
        all_results = analysis.trial_dataframes
       
        plt.figure(figsize=(12, 6))
        for trial in all_results.values():
            plt.plot(trial[f"{x_axis}"], trial[f"{y_axis}"], marker='o', label=f"Trial {trial['trial_id'][0]}")

        plt.title(f'{y_axis} vs {x_axis}')
        plt.xlabel(f"{x_axis}")
        plt.ylabel(f"{y_axis}")
        plt.legend()
        plt.grid(True)
        plt.show()