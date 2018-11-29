import numpy as np
import sklearn
import sklearn.datasets

from neural_network import NeuralNetwork


def load_planar_dataset():
    np.random.seed(1)
    m = 400  # number of examples
    N = int(m / 2)  # number of points per class
    D = 2  # dimensionality
    X = np.zeros((m, D))  # data matrix where each row is a single example
    Y = np.zeros((m, 1), dtype='uint8')  # labels vector (0 for red, 1 for blue)
    a = 4  # maximum ray of the flower

    for j in range(2):
        ix = range(N * j, N * (j + 1))
        t = np.linspace(j * 3.12, (j + 1) * 3.12, N) + np.random.randn(N) * 0.2  # theta
        r = a * np.sin(4 * t) + np.random.randn(N) * 0.2  # radius
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        Y[ix] = j

    X = X.T
    Y = Y.T

    return X, Y


if __name__ == '__main__':
    entries, outputs = load_planar_dataset()
    #
    # entries = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    # outputs = np.array([0, 1, 1, 0])


    shape_X = entries.shape
    shape_Y = outputs.shape
    num_examples = outputs.shape[1]  # training set size
    ### END CODE HERE ###

    print(entries)
    print('The shape of X is: ' + str(shape_X))
    print(outputs)
    print('The shape of Y is: ' + str(shape_Y))
    print('I have m = %d training examples!' % (num_examples))

    num_inputs = shape_X[0]
    num_hidden = 4
    num_outputs = shape_Y[0]

    NN = NeuralNetwork(0.24, num_inputs, num_hidden, num_outputs, entries, outputs, num_examples)

    NN.train()



