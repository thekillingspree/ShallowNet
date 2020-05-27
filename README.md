# Shallow Neural Network

A shallow neural network is a neural network with only 1 hidden layer.

This project is tries to implement a Shallow Neural network using only numpy. 

This shallow neural network currently supports binary classification only. 

The hidden layers use the [**tanh**](https://ml-cheatsheet.readthedocs.io/en/latest/activation_functions.html#tanh) activation function and the output layer uses the [**sigmoid**](https://ml-cheatsheet.readthedocs.io/en/latest/activation_functions.html#sigmoid) activation function.

# Usage

For detailed usage guide, refer [Example.ipynb](https://github.com/)

    shallow = ShallowNet(hiddenLayerSize=5)
    shallow.fit(x_train, y_train, learning_rate=0.01, 
    verbose=True, iterations=10000, random_state=3)

Evaluate Model

    shallow.evaluate(x_test, y_test)

Save Model in hdf5 format

    shallow.save_model('model.h5')

Load Model

    shallow.load_model('model.h5')