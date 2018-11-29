import numpy as np

class NeuralNetwork:
    def __init__(self, reg_factor, num_inputs, num_hidden, num_outputs, entries, outputs, num_examples):
        # Definição dos parâmetros
        self.reg_factor = reg_factor
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs
        self.entries = entries
        self.outputs = outputs
        self.num_examples = num_examples
        self.epochs = 10000
        self.learning_rate = 1.2
        self.momentum = 1

    # Define pesos aleatórios para a rede
    def init_random_weights(self):
        np.random.seed(2)
        # Camada de entrada para camada oculta
        self.W1 = np.random.randn(self.num_hidden, self.num_inputs) / 100
        # Bias da primeira camada
        self.b1 = np.zeros((self.num_hidden, 1))
        self.b1 = np.array([[1.74481176],
                            [-0.7612069],
                            [0.3190391],
                            [-0.24937038]])
        # Camada oculta para camada de saída
        self.W2 = np.random.randn(self.num_outputs, self.num_hidden) / 100
        # Bias da segunda camada
        self.b2 = np.zeros((self.num_outputs, 1))
        self.b2 = np.array([[-1.3]])

    # Função de soma
    def calculate_sum(self, weights, entries):
        sum = np.dot(weights, entries)
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
        self.sum_entries = self.calculate_sum(self.W1, entries) + self.b1
        self.activations_hidden_layer = np.tanh(self.sum_entries)
        self.sum_hidden = self.calculate_sum(self.W2, self.activations_hidden_layer) + self.b2
        self.activation_output_layer = self.calculate_sigmoid(self.sum_hidden)
        # print(np.mean(self.sum_entries), np.mean(self.activations_hidden_layer), np.mean(self.sum_hidden), np.mean(self.activation_output_layer))

    def calculate_cost(self):
        logprobs = np.multiply(np.log(self.activation_output_layer), self.outputs) + np.multiply((1 - self.outputs), np.log(1 - self.activation_output_layer))
        cost = - np.sum(logprobs) / self.num_examples
        return cost

    def backward_propagation(self):

        dZ2 = self.activation_output_layer - self.outputs
        self.derivatives_W2 = (1 / self.num_examples) * np.dot(dZ2, self.activations_hidden_layer.T)
        self.derivatives_b2 = (1 / self.num_examples) * np.sum(dZ2, axis=1, keepdims=True)
        dZ1 = np.multiply(np.dot(self.W2.T, dZ2), 1 - np.power(self.activations_hidden_layer, 2))
        self.derivatives_W1 = (1 / self.num_examples) * np.dot(dZ1, self.entries.T)
        self.derivatives_b1 = (1 / self.num_examples) * np.sum(dZ1, axis=1, keepdims=True)

    def update_parameters(self):

        self.W1 -= self.learning_rate * self.derivatives_W1
        self.b1 -= self.learning_rate * self.derivatives_b1
        self.W2 -= self.learning_rate * self.derivatives_W2
        self.b2 -= self.learning_rate * self.derivatives_b2


    def train(self):
        self.init_random_weights()
        for i in range (self.epochs):
            self.forward_propagation(self.entries)
            cost = self.calculate_cost()
            self.backward_propagation()
            self.update_parameters()

            # Print the cost every 1000 iterations
            if i % 1000 == 0:
                print("Cost after iteration %i: %f" % (i, cost))


        # errors = self.calculate_output_error(self.outputs, output)
        # self.absolute_mean = np.mean(np.abs(errors))
        # output_derivative = self.calculate_derivative_sigmoid(output)
        # output_delta = self.calculate_output_delta(errors, output_derivative)
        # hidden_derivative = self.calculate_derivative_sigmoid(self.activations_hidden_layer)
        # hidden_delta = self.calculate_hidden_delta(hidden_derivative, self.W2, output_delta)
        # self.W2 = self.calculate_new_weights(self.W2, self.activations_hidden_layer, output_delta)
        # self.W1 = self.calculate_new_weights(self.W1, self.entries, hidden_delta)

        # print(self.W1)
        # print("------")
        # print(self.W2)




