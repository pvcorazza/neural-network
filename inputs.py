import copy
import csv

import numpy as np


def read_data(filename):
    if filename == "benchmark.csv":
        data = list(csv.reader(open("data/" + filename, "r"), delimiter=";"))
        return data

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


def read_network_file(name):

    try:
        file = open(name, "r")
    except:
        exit("Error while opening \"network.txt\"")

    reg_factor = float(file.readline())
    network_info = file.readlines()
    network_info = [int(i) for i in network_info]

    return reg_factor, network_info

def read_dataset_file(name):
    """
    Faz a leitura de arquivos do tipo dataset.txt.
    Retorna uma lista de instâncias de treinamento, na qual cada instância é
    uma tupla de duas listas: atributos e saídas esperadas.
    """

    try:
        file = open(name, "r")
    except:
        exit("Error while opening \"dataset.txt\"")

    lines = (line.replace(';',',') for line in file)
    data = np.genfromtxt(lines, dtype=float, delimiter=',')

    outputs = data[:, -1]
    entries = np.delete(data, -1, axis=1)

    return entries.astype(np.float), np.reshape(outputs.astype(int), (-1, 1))


