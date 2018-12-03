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

    # entries = feature_normalization(entries)

    np.set_printoptions(precision=5,suppress=True)
    print("------------------------------------------------------")
    print("----  INF01017 - Aprendizado de Máquina (2018/2)  ----")
    print("---- Trabalho 2 - Redes Neurais (Backpropagation) ----")
    print("----             Paulo Victor Corazza             ----")
    print("------------------------------------------------------")
    print("1. Inicializando rede neural")
    print("* Estrutura: " + str(structure))
    print("* Parâmetro lambda: " + str(lb))
    print()
    print("* Pesos: ")

    for i in range(1, len(structure)):
        print("\tW" + str(i) + " = " + str(weights["W" + str(i)]).replace('\n','\n\t\t '))

        print("\tb" + str(i) + " = " + str(weights["b" + str(i)]).replace('\n','\n\t\t '))

    print()
    print("* Conjunto de treinamento normalizado: ")
    print("\tEntradas = "+str(entries).replace('\n','\n\t\t\t   '))
    print("\tSaídas =   "+str(outputs).replace('\n','\n\t\t\t   '))
    print()

    NN = NeuralNetwork(lb, entries.T, outputs.T, structure, iterations=100000, learning_rate=0.1, weights=weights)
    grads, approx = NN.get_gradients_to_compare(entries.T)
    NN.compare_gradients(grads,approx)

    print("Salvando gradientes do backpropagation no arquivo \"gradients_backpropagation.txt\"...")

    save_gradients(grads,"gradients_backpropagation.txt")
    # save_gradients(approx,"gradients_numeric.txt", numeric=True)

    exit()




