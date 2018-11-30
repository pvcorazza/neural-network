import numpy as np

class NeuralNetwork:
    def __init__(self, reg_factor, entries, outputs, layers, learning_rate=1.2, epochs = 1000000):

        # Definição dos parâmetros
        self.reg_factor = reg_factor
        self.entries = entries
        self.outputs = outputs
        self.layers = layers
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.num_examples = np.shape(self.outputs)[1]
        self.parameters = {}
        # self.momentum = 1
        self.J = []

    # Define pesos aleatórios para a rede
    def init_random_weights(self):
        for l in range(1, len(self.layers)):
            # Pesos
            self.parameters['W' + str(l)] = 2 * np.random.randn(self.layers[l], self.layers[l - 1]) - 1
            # Bias
            self.parameters['b' + str(l)] = np.ones(shape=(self.layers[l], 1))

    # Função sigmóide
    def calculate_sigmoid(self, sum):
        result = 1 / (1 + np.exp(-sum))
        return result

    # Derivada da função sigmóide
    def calculate_derivative_sigmoid(self, sig_result):
        result = sig_result * (1 - sig_result)
        return result

    # # Função de propagação
    # def forward_propagation(self, entries):
    #     # 1.al=1 = x(i)
    #     self.a1 = entries
    #
    #     # 1.z(l=k) = θ(l=k-1) a(l=k-1)
    #     z2 = np.dot(self.W1, self.a1) + self.b1
    #
    #     # 2.a(l=k) = g(z(l=k))
    #     self.a2 = np.tanh(z2)
    #
    #     # 4.z(l=L) = θ(l=L-1) a(l=L-1)
    #     z3 = np.dot(self.W2, self.a2) + self.b2
    #
    #     # 5. a(l=L) = g(z(l=L))
    #     self.a3 = self.calculate_sigmoid(z3)
    #
    #     # 6.Retorna fθ(x(i)) = a(l=L)
    #     return self.a3

    def forward_propagation(self):
        caches = []
        A = self.entries
        L = len(self.parameters) // 2  # number of layers in the neural network

        for l in range(1, L + 1):
            Z = self.parameters['W' + str(l)].dot(A) + self.parameters['b' + str(l)]
            linear_cache = (A, self.parameters['W' + str(l)], self.parameters['b' + str(l)])
            A = self.calculate_sigmoid(Z)
            cache = (linear_cache, Z)
            caches.append(cache)

        return A, caches

    # Necessário inserir a regularização
    def calculate_cost(self, AL):
        cost = - np.sum(np.multiply(np.log(AL), self.outputs) + np.multiply((1 - self.outputs), np.log(1 - AL))) / self.num_examples
        return cost

    def L_model_backward(self, AL, caches):
        """
        Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group

        Arguments:
        AL -- probability vector, output of the forward propagation (L_model_forward())
        Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
        caches -- list of caches containing:
                    every cache of linear_activation_forward() with "relu" (there are (L-1) or them, indexes from 0 to L-2)
                    the cache of linear_activation_forward() with "sigmoid" (there is one, index L-1)

        Returns:
        grads -- A dictionary with the gradients
                 grads["dA" + str(l)] = ...
                 grads["dW" + str(l)] = ...
                 grads["db" + str(l)] = ...
        """
        grads = {}

        dAL = - (np.divide(self.outputs, AL) - np.divide(1 - self.outputs, 1 - AL))

        linear_cache, activation_cache = caches[len(caches) - 1]

        dZ = dAL * self.calculate_derivative_sigmoid(self.calculate_sigmoid(activation_cache))

        A_prev, W, b = linear_cache
        m = A_prev.shape[1]

        grads["dW" + str(len(caches))] = 1. / m * np.dot(dZ, A_prev.T)
        grads["db" + str(len(caches))] = 1. / m * np.sum(dZ, axis=1, keepdims=True)
        grads["dA" + str(len(caches) - 1)] = np.dot(W.T, dZ)

        for l in reversed(range(len(caches) - 1)):
            linear_cache, activation_cache = caches[l]
            dZ = grads["dA" + str(l + 1)] * self.calculate_derivative_sigmoid(self.calculate_sigmoid(activation_cache))

            A_prev, W, b = linear_cache
            m = A_prev.shape[1]

            grads["dW" + str(l + 1)] = 1. / m * np.dot(dZ, A_prev.T)
            grads["db" + str(l + 1)] = 1. / m * np.sum(dZ, axis=1, keepdims=True)
            grads["dA" + str(l)] = np.dot(W.T, dZ)

        return grads

    # def backward_propagation(self):
    #
    #     # Vetor com gradientes dos pesos
    #     self.dw = {}
    #
    #     # Vetor com gradientes de bias
    #     self.db = {}
    #
    #     # Calcular o valor de 𝛿 para os neurônios da camada de saída:
    #     # 1.2.𝛿(l=L) = fθ(x(i)) - y(i)
    #     delta3 = self.a3 - self.outputs
    #
    #     #1.3.Para cada camada k=L-1…2 // calcula os deltas para as camadas ocultas
    #     #𝛿(l=k) = [θ(l=k)]T 𝛿(l=k+1) .* a(l=k) .* (1-a(l=k))
    #     delta2 = np.multiply(np.dot(self.W2.T, delta3), 1 - np.power(self.a2, 2))
    #
    #     # Armazena gradientes
    #     self.dw[2] = (1 / self.num_examples) * np.dot(delta3, self.a2.T)
    #     self.db[2] = (1 / self.num_examples) * np.sum(delta3, axis=1, keepdims=True)
    #     self.dw[1] = (1 / self.num_examples) * np.dot(delta2, self.entries.T)
    #     self.db[1] = (1 / self.num_examples) * np.sum(delta2, axis=1, keepdims=True)


    def update_parameters(self, grads):
        # Update rule for each parameter. Use a for loop.
        for l in range(len(self.parameters) // 2):
            self.parameters["W" + str(l + 1)] -= self.learning_rate * grads["dW" + str(l + 1)]
            self.parameters["b" + str(l + 1)] -= self.learning_rate * grads["db" + str(l + 1)]

    def train(self):

        # np.random.seed(1)


        # Inicia pesos aleatórios
        self.init_random_weights()

        # Loop (gradient descent)
        for i in range(0, self.epochs):

            # Propagar o exemplo pela rede, calculando sua saída fθ(x)
            AL, caches = self.forward_propagation()

            # Calcula custo
            cost = self.calculate_cost(AL)
            erroCamadaSaida = AL - self.outputs
            mediaAbsoluta = np.mean(np.abs(erroCamadaSaida))

            # Backward propagation.
            grads = self.L_model_backward(AL, caches)

            # Atualiza parâmetros.
            self.update_parameters(grads)

            # Print the cost every 100 training example
            if i % 1000 == 0:
                print("Cost after iteration %i: %f" % (i, cost))
                self.J.append(cost)
                # print(AL)
                print("Erro: %.8f" % mediaAbsoluta)

        return self.parameters







