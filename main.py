import sys

from inputs import *
from neural_network import NeuralNetwork


def gradient_check_n_test_case():
    np.random.seed(1)
    x = np.random.randn(4, 3)
    y = np.array([1, 1, 0])
    W1 = np.random.randn(5, 4)
    b1 = np.random.randn(5, 1)
    W2 = np.random.randn(3, 5)
    b2 = np.random.randn(3, 1)
    W3 = np.random.randn(1, 3)
    b3 = np.random.randn(1, 1)
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3}

    return x, y, parameters

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
    #
    # X, Y, parameters = gradient_check_n_test_case()
    #
    # layers_dims = (4, 5, 3, 1)
    #
    # NN = NeuralNetwork(0, X, Y, layers_dims, epochs=5000,weights=parameters)
    #
    #
    # NN.train()
    #
    #
    # # a = NN.forward_propagation(X, parameters)
    # # b = NN.backward_propagation()
    # # exit()
    #
    # gradients = ba(X, Y, cache)
    # difference = gradient_check_n(parameters, gradients, X, Y)


    entries = feature_normalization(entries)
    entries_training, entries_test = divide_train_test(entries, 0.75)
    outputs_training, outputs_test = divide_train_test(outputs, 0.75)

    layers_dims = (entries_training.shape[1], 4, outputs_training.shape[1])

    NN = NeuralNetwork(1.2, entries_training.T, outputs_training.T, layers_dims, iterations=10000)

    grads, approx = NN.get_gradients_to_compare(entries_training.T)

    NN.compare_gradients(grads,approx)
    exit()

    weights, gradients = NN.train()

    print("Training")
    NN.predict2(entries_training.T, outputs_training.T)

    print("Test")
    NN.predict2(entries_test.T, outputs_test.T)

    exit()





