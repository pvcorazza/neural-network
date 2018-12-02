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

    entries_training, entries_test = divide_train_test(entries, 0.75)

    outputs_training, outputs_test = divide_train_test(outputs, 0.75)

    entries_training, entries_test = get_transpose(entries_training, entries_test)
    outputs_training, outputs_test = get_transpose(outputs_training, outputs_test)




    layers_dims = (entries_training.shape[0], 4, outputs_training.shape[0])

    NN = NeuralNetwork(0.4, entries_training, outputs_training, layers_dims, epochs=10000)

    NN.train()

    print("TRAINING")
    NN.predict2(entries_training, outputs_training)
    print("TEST")
    NN.predict2(entries_test, outputs_test)





