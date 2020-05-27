import numpy as np
import time
import h5py

class ShallowNet:
    def __init__(self, hiddenLayerSize=4):
        self.hiddenLayerSize = hiddenLayerSize
        self.parameters = {}
        self.gradients = {}

    def cross_entropy(self, A2):

        m = self.y.shape[1]

        log = np.multiply(np.log(A2), self.y) + np.multiply(np.log(1-A2), 1-self.y)

        return (-1/m) * np.sum(log)


    def sigmoid(self, z):
        return 1 / (1 + np.exp(-1 * z))

    def __init_parameters(self):
        np.random.seed(self.random_state)
        self.parameters['W1'] = np.random.randn(self.hiddenLayerSize, self.X.shape[0]) * 0.01
        self.parameters['b1'] = np.zeros((self.hiddenLayerSize, 1))
        self.parameters['W2'] = np.random.randn(self.y.shape[0], self.hiddenLayerSize) * 0.01
        self.parameters['b2'] = np.zeros((self.y.shape[0], 1))

    def __forward_propagation(self, X):

        Z1 = np.dot(self.parameters['W1'], X) + self.parameters['b1']
        A1 = np.tanh(Z1)
        Z2 = np.dot(self.parameters['W2'], A1) + self.parameters['b2']
        A2 = self.sigmoid(Z2)
        return {
            'Z1': Z1,
            'A1': A1,
            'Z2': Z2,
            'A2': A2,
        }

    def __backpropagate(self, forward_vals):
        
        m = self.y.shape[1]

        dZ2 = forward_vals['A2'] - self.y
        dW2 = (1/ m) * np.dot(dZ2, forward_vals['A1'].T)
        db2 = (1/ m) * np.sum(dZ2, axis=1, keepdims=True)
        dZ1 = np.multiply(np.dot(self.parameters['W2'].T, dZ2), 1-(forward_vals['A1'] ** 2))
        dW1 = (1/ m) * np.dot(dZ1, self.X.T)
        db1 = (1/ m) * np.sum(dZ1, axis=1, keepdims=True)

        self.gradients = {
            'dW1': dW1,
            'db1': db1,
            'dW2': dW2,
            'db2': db2
        }

    def __update_parameters(self):

        for i in ['W1', 'b1', 'W2', 'b2']:
            self.parameters[i] = self.parameters[i] - (self.learning_rate * self.gradients[f'd{i}'])



    def save_model(self, path):
        with h5py.File(path, 'w') as model_file:
            for key, value in self.parameters.items():
                model_file.create_dataset(key, data=value)

    def load_model(self, path):
        with h5py.File(path, 'r') as model:
            for key in model.keys():
                self.parameters[key] = model[key][:]

    def fit(self, X, y, learning_rate=0.01, verbose=True, iterations=10000, random_state=2):
        self.X = X
        self.y = y
        self.learning_rate = learning_rate
        self.random_state = random_state

        self.__init_parameters()
        for i in range(iterations):
            start = time.time()
            forward_vals = self.__forward_propagation(X)
            cost = self.cross_entropy(forward_vals['A2'])
            self.__backpropagate(forward_vals)
            self.__update_parameters()
            end = time.time()

            diff = end - start

            prog = int((i / iterations)  * 10) + 1
            eta = diff * (iterations - i)

            if verbose and (i % 10 == 0 or i == iterations - 1):
                print(f"\rProgress: {'❚' * (prog) + '-'*(10-prog)} cost={format(cost, '.6f')} ETA={format(eta, '.2f')} seconds", flush=True, end=('\n' if i == iterations-1 else '\r'))


    def predict(self, X):
        forward_vals = self.__forward_propagation(X)
        return (forward_vals['A2'] > 0.5)

    def evaluate(self, X, y):
        preds = self.predict(X)
        acc = float(np.dot(y, preds.T) + np.dot(1-y, 1-preds.T)) / y.size
        return acc * 100
