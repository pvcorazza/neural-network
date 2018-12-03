import copy
import csv

import numpy as np

def read_data(filename):

    data = list(csv.reader(open("data/" + filename, "r"), delimiter=","))
    if not data:
        return None

    # Breast Cancer Wisconsin (32 atributos, 569 exemplos, 2 classes)
    # https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)
    # Objetivo: predizer se um determinado exame médico indica ou não a presença de câncer.
    if filename == "wdbc.data":

        # 1) ID number
        # 2) Diagnosis (M = malignant, B = benign)
        # 3-32)
        #
        # Ten real-valued features are computed for each cell nucleus:
        #
        # 	a) radius (mean of distances from center to points on the perimeter)
        # 	b) texture (standard deviation of gray-scale values)
        # 	c) perimeter
        # 	d) area
        # 	e) smoothness (local variation in radius lengths)
        # 	f) compactness (perimeter^2 / area - 1.0)
        # 	g) concavity (severity of concave portions of the contour)
        # 	h) concave points (number of concave portions of the contour)
        # 	i) symmetry
        # 	j) fractal dimension ("coastline approximation" - 1)

        # Remove os valores para o ID
        for x in data:
            del x[0]


        data = np.array(data)

        outputs = data[:, 0]
        entries = np.delete(data, 0, axis=1)

        for n, i in enumerate(outputs):
            if i == "M":
                outputs[n] = 1
            else:
                outputs[n] = 0

        return entries.astype(np.float), np.reshape(outputs.astype(int), (-1, 1))

    # Wine Data Set (13 atributos, 178 exemplos, 3 classes)
    # https://archive.ics.uci.edu/ml/datasets/wine
    # Objetivo: predizer o tipo de um vinho baseado em sua composição química
    if filename == "wine.data":

        attributes = ['Class', 'Alcohol', 'Malic', 'Ash', 'Alcalinity', 'Magnesium', 'Phenols',
                      'Flavanoids', 'Nonflavanoid', 'Proanthocyanins', 'Color', 'Hue', 'Od', 'Proline']


        for i in range(len(data)):
            for j in range (len(data[0])):
                if data[i][j][0] == ".":
                    data[i][j] = copy.deepcopy("0"+data[i][j])

        data.insert(0, attributes)



        # Posiciona a classe ao final dos dados para padronização
        for x in data:
            x.append(copy.deepcopy(x[0]))
            del x[0]

        return data

    # 3. Ionosphere Data Set (34 atributos, 351 exemplos, 2 classes)
    # https://archive.ics.uci.edu/ml/datasets/Ionosphere
    # Objetivo: predizer se a captura de sinais de um radar da ionosfera é adequada para
    # análises posteriores ou não (’good’ ou ’bad’)
    if filename == "ionosphere.data":

        attributes = []

        for i in range(34):
            attributes.append("Signal" + str(i))

        attributes.append("Class")

        data.insert(0, attributes)

        return data

# Leitura do arquivo com informações sobre a rede neural
def read_network_file(name):

    try:
        file = open(name, "r")
    except:
        exit("Error while opening \"network.txt\"")

    reg_factor = float(file.readline())
    network_info = file.readlines()
    network_info = tuple([int(i) for i in network_info])

    return reg_factor, network_info

# Leitura do arquivo de pesos
def read_weights_file(name):

    with open(name, 'r') as file:
        lines = file.readlines()

    weights = {}
    i=1
    for line in lines:

        layer = [neuron.split(',') for neuron in line.split(';')]
        weights_array = np.asarray([[float(weight) for weight in entry] for entry in layer])
        bias = weights_array[:, 0]
        layer_weight = np.delete(weights_array, 0, axis=1)
        weights['W' + str(i)] = layer_weight
        weights['b' + str(i)] = np.reshape(bias,(len(bias),1))
        i =i+1

    return weights


# Leitura do arquivo dataset
def read_dataset_file(name):

    try:
        file = open(name, "r")
    except:
        exit("Error while opening \"dataset.txt\"")


    with open(name, 'r') as file:
        lines = file.readlines()

    entries = []
    output = []

    for line in lines:
        layer = [neuron.split(',') for neuron in line.split(';')]
        weights_array = np.asarray([[float(weight) for weight in entry] for entry in layer])
        entries.append(weights_array[0])
        output.append((weights_array[1]))


    return np.asarray(entries), np.asarray(output)



# Normalização das features
def feature_normalization(data):
    max = np.max(data, axis=0)
    min = np.min(data, axis=0)

    result = (data - min) / (max - min)

    return result

# Função que retorna as transpostas
def get_transpose (entries, outputs):

    return entries.T, outputs.T

# Divide dado em conjuntos de treinamento e teste
def divide_train_test(data, factor):
    training, test = data[:round(len(data) * factor), :], data[round(len(data) * factor):, :]
    return training, test

def save_gradients(gradients, name, numeric=False):
    np.set_printoptions(formatter={'float': lambda x: "{0:0.5f}".format(x)})

    with open(name, 'w') as file:
        for i in range(1, (len(gradients)//2) + 1):
            if numeric:
                array = np.append(gradients["b" + str(i)], gradients["W" + str(i)], axis=1)
            else:
                array = np.append(gradients["db" + str(i)], gradients["dW" + str(i)], axis=1)
            line=""
            for array_line in array:
                array_line = np.array2string(array_line, precision=5, separator=', ')
                line = "".join([line,array_line[1:-1]]) + "; "
            file.write(line[:-2]+"\n")




