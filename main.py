import numpy as np

from neural_network import NeuralNetwork

if __name__ == '__main__':
    entries = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    outputs = np.array([[0], [1], [1], [0]])

    NN = NeuralNetwork(0.24, 2, 3, 1, entries, outputs)

    NN.train()

    output = NN.forward_propagation(entries)
    print(output)



