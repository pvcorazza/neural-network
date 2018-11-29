import numpy as np

class NeuralNetwork:
    def __init__(self, reg_factor, num_inputs, num_hidden, num_outputs, entries, outputs):
        # Definição dos parâmetros
        self.reg_factor = reg_factor
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs
        self.entries = entries
        self.outputs = outputs
        self.epochs = 100000
        self.learning_rate = 0.6
        self.momentum = 1

    # Define pesos aleatórios para a rede
    def init_random_weights(self):
        # Camada de entrada para camada oculta
        self.W1 = np.random.randn(self.num_inputs, self.num_hidden)
        # Camada oculta para camada de saída
        self.W2 = np.random.randn(self.num_hidden, self.num_outputs)

    # Função de soma
    def calculate_sum(self, entries, weights):
        sum = np.dot(entries, weights)
        return sum

    # Função sigmóide
    def calculate_sigmoid(self, sum):
        result = 1 / (1 + np.exp(-sum))
        return result

    # Derivada da função sigmóide
    def calculate_derivative_sigmoid(self, sig_result):
        result = sig_result * (1-sig_result)
        return result

    # Erro na saida
    def calculate_output_error(self, outputs, result_output_layer):
        result = outputs - result_output_layer
        return result

    # Delta da saida
    def calculate_output_delta(self, output_error, sigmoid_derivative):
        result = output_error * sigmoid_derivative
        return result

    # Delta da camada oculta
    def calculate_hidden_delta(self, sigmoid_derivative, weight, output_delta):
        result = np.dot(output_delta, weight.T) * sigmoid_derivative
        return result

    # Calcula novos pesos para camada
    def calculate_new_weights(self, weights, layer, delta):
        pesosNovo0 = np.dot(layer.T, delta)
        weights = (weights * self.momentum) + ( pesosNovo0 * self.learning_rate)
        return weights

    # Função de propagação
    def forward_propagation(self, entries):
        sum_entries = self.calculate_sum(entries, self.W1)
        self.activations_hidden_layer = self.calculate_sigmoid(sum_entries)
        sum_hidden = self.calculate_sum(self.activations_hidden_layer, self.W2)
        result_output_layer = self.calculate_sigmoid(sum_hidden)
        return result_output_layer

    def train(self):
        self.init_random_weights()
        for j in range (self.epochs):
            output = self.forward_propagation(self.entries)
            errors = self.calculate_output_error(self.outputs, output)
            self.absolute_mean = np.mean(np.abs(errors))
            output_derivative = self.calculate_derivative_sigmoid(output)
            output_delta = self.calculate_output_delta(errors, output_derivative)
            hidden_derivative = self.calculate_derivative_sigmoid(self.activations_hidden_layer)
            hidden_delta = self.calculate_hidden_delta(hidden_derivative, self.W2, output_delta)
            self.W2 = self.calculate_new_weights(self.W2, self.activations_hidden_layer, output_delta)
            self.W1 = self.calculate_new_weights(self.W1, self.entries, hidden_delta)

            # print(self.W1)
            # print("------")
            # print(self.W2)




