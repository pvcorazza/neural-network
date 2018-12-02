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

    # X, Y, parameters = gradient_check_n_test_case()
    #
    # Y = np.reshape(Y, (3,1))
    #
    # layers_dims = (X.shape[0], 5,3, Y.shape[0])
    #
    # NN = NeuralNetwork(0, X, Y, layers_dims, epochs=5000)
    #
    # a, cache = NN.forward_propagation(X, parameters)
    #
    # exit()
    #
    #
    #
    #
    # exit()

    X, Y, parameters = gradient_check_n_test_case()

    layers_dims = (4, 5, 3, 1)

    NN = NeuralNetwork(0, X, Y, layers_dims, epochs=5000,weights=parameters)


    NN.train()


    # a = NN.forward_propagation(X, parameters)
    # b = NN.backward_propagation()
    # exit()

    gradients = ba(X, Y, cache)
    difference = gradient_check_n(parameters, gradients, X, Y)



    # entries = feature_normalization(entries)
    # entries, outputs = get_transpose(entries,outputs)
    #
    # entries_training, entries_test = divide_train_test(entries, 0.75)
    #
    # outputs_training, outputs_test = divide_train_test(outputs, 0.75)
    #
    # entries_training, entries_test = get_transpose(entries_training, entries_test)
    # outputs_training, outputs_test = get_transpose(outputs_training, outputs_test)


    # layers_dims = (entries.shape[1], 4, outputs.shape[1])
    #
    # NN = NeuralNetwork(0, entries.T, outputs.T, layers_dims, epochs=5000)
    #
    # NN.train()

    # print("TRAINING")
    # NN.predict2(entries_training, outputs_training)
    # print("TEST")
    # NN.predict2(entries_test, outputs_test)

    # print(NN.forward_propagation(entries, NN.weights))
    # exit()
    #
    # NN.gradient_check_n()






