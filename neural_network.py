import numpy as np


class NeuralNetwork:
    def __init__(self, reg_factor, num_inputs, num_hidden, num_outputs):
        # Definição dos parâmetros
        self.reg_factor = reg_factor
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs

    # Define pesos aleatórios para a rede
    def init_random_weights(self):

        # self.W1 = np.array([[-0.424, -0.740, -0.961], [0.358, -0.577, -0.469]])
        # self.W2 = np.array([[-0.017], [-0.893], [0.148]])

        # Camada de entrada para camada oculta
        self.W1 = np.random.randn(self.num_inputs, self.num_hidden)
        # Camada oculta para camada de saída
        self.W2 = np.random.randn(self.num_hidden, self.num_outputs)

    # Função de soma
    def sum_function(self, entries, weights):
        sum = np.dot(entries, weights)
        return sum

    # Função sigmóide
    def sigmoid_function(self, sum):
        result = 1 / (1 + np.exp(-sum))
        return result

    # Derivada da função sigmóide
    def derivative_sigmoid_function(self, sig_result):
        result = sig_result * (1-sig_result)
        return result

    # Função de propagação
    def propagation(self, entries):
        sum_entries = self.sum_function(entries, self.W1)
        activations_hidden_layer = self.sigmoid_function(sum_entries)
        sum_hidden = self.sum_function(activations_hidden_layer, self.W2)
        output_layer = self.sigmoid_function(sum_hidden)
        return output_layer

