import numpy as np

from neural_network import NeuralNetwork

if __name__ == '__main__':
    # entries = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    # outputs = np.array([[0], [1], [1], [0]])

    entries = np.array([[3, 5], [5, 1], [10, 2]])

    epochs = 100

    NN = NeuralNetwork(0.24, 2, 3, 1)
    NN.init_random_weights()

    # error_output_layer = outputs - output_layer
    #
    # absolute_mean = np.mean(np.abs(error_output_layer))
    #
    # print(absolute_mean)