import sys

import numpy as np

from inputs import read_network_file, read_data, read_dataset_file
from neural_network import NeuralNetwork

if __name__ == '__main__':

    if(sys.argv[1] == "network.txt"):
        lb, structure = read_network_file(sys.argv[1])
    else:
        sys.exit('Error! The input format is: ./backpropagation network.txt initial_weights.txt dataset.txt')
    if (sys.argv[2] == "dataset.txt"):
        entries, outputs = read_dataset_file(sys.argv[2])
    else:
        sys.exit('Error! The input format is: ./backpropagation network.txt initial_weights.txt dataset.txt')

    np.set_printoptions(suppress=True)

    entries = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0]])
    outputs = np.array([[0], [1], [1], [0]])

    layers_dims = (entries.shape[1], 7, 4, 5, outputs.shape[1])

    entries = entries.T
    outputs = outputs.T

    NN = NeuralNetwork(0.24, entries, outputs, layers_dims, epochs=10000)

    NN.train()

    entry = np.array([[0, 0, 0]]).T
    entry2 = np.array([[1, 0, 0]]).T

    print(NN.accuracy(entry2, [[1]]))



