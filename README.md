# Ad-DL-Midterm-Project

## Dataset

We have the Fashion MNIST which is also a large database on clothing apparels. Our work is to use three different optimizer algorithms (excluding vanilla GD/SGD) to figure out the details, pros, and cons of each one by applying all three algorithms to the artificial neural network (ANN) model to identify clothing accessories.

## Deep Learning Model - ANN

Artificial Neural network resembles the brain's neural network with densely connected neurons in between input and output layers. It has a hidden layers where the internal processing happens in ANN. The neural network's objective is to minimise the loss(actual-predicted) by using the learning method called as back propogation where the weights get re-initialized in each connecting layer for many epochs through which the loss is minimised.

We will build and compile an ANN model without any hyperparameter tuning for the model accuracy

**Defining the model**
To define the model we need the Sequential() function which helps us to build the base neural network on that we have to decide the dense layers and neurons.

- We have used relu activation function for the hidden layers and softmax for the output layer
- Since we didn't normalize our dataset we are using BatchNormalization() function to normalize in the neural network
- We are also considering the drop out layer in each hidden layers to reduce the chances of overfitting

**Compiling the model**
The base model of neural network is ready. It's time to connect the brain for the neural network. In this part we tell the neural network on how to learn the model where we signify the type of loss function and which optimizer and metrics to use.

**Fitting the model**
Now, its time to train our neural network. Since we didn't use any hyperparameter training we used our own values for batch size and epochs.

Our project includes one Python (.py) file containing model functions for training, testing, evaluating etc.

Additionally, I provide three Jupyter Notebook (.ipynb) files to facilitate your work.

- The first 3 notebook, named **fashion_mnist_optimizer*.ipynb**, is specifically set up for experimenting with different optimizers to understand their impact on model performance, each person takes one.

- The last notebook, **fashion_mnist_model_test.ipynb**, focuses on the accuracy of models constructed It is important to note that the evaluation in this notebook is conducted without applying any optimizers, providing a baseline performance metric for each model configuration.
