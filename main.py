import sys

from inputs import *
from neural_network import NeuralNetwork

if __name__ == '__main__':

    if(sys.argv[1] == "network.txt"):
        lb, structure = read_network_file(sys.argv[1])
    else:
        sys.exit('Error! The input format is: ./backpropagation network.txt initial_weights.txt dataset.txt')
    if (sys.argv[2] == "weights.txt"):
        weights = read_weights_file(sys.argv[2])
    else:
        sys.exit('Error! The input format is: ./backpropagation network.txt initial_weights.txt dataset.txt')
    if (sys.argv[3] == "dataset.txt"):
        entries, outputs = read_dataset_file(sys.argv[3])
    else:
        sys.exit('Error! The input format is: ./backpropagation network.txt initial_weights.txt dataset.txt')

    np.set_printoptions(suppress=True)


    entries = feature_normalization(entries)
    entries, outputs = get_transpose(entries, outputs)

    layers_dims = (entries.shape[0], 5, outputs.shape[0])

    NN = NeuralNetwork(0.24, entries, outputs, layers_dims, epochs=10000)

    NN.train()

    #
    # entry = np.array([[0, 0, 0]]).T
    # entry2 = np.array([[1, 0, 0]]).T

    # print(NN.accuracy(entries, outputs))



