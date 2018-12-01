import numpy as np

class NeuralNetwork:
    def __init__(self, reg_factor, entries, outputs, layers, learning_rate=1.2, epochs = 1000000, weights=None):

        # Definição dos parâmetros
        self.reg_factor = reg_factor
        self.entries = entries
        self.outputs = outputs
        self.layers = layers
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.num_examples = np.shape(self.outputs)[1]
        self.weights = weights
        # self.momentum = 1
        self.J = []
        self.gradients = {}

    # Define pesos aleatórios para a rede
    def init_weights(self):
        if (self.weights == None):
            self.weights = {}
            for l in range(1, len(self.layers)):
                # Pesos
                self.weights['W' + str(l)] = 2 * np.random.randn(self.layers[l], self.layers[l - 1]) - 1
                # Bias
                self.weights['b' + str(l)] = np.ones(shape=(self.layers[l], 1))

    # Função sigmóide
    def calculate_sigmoid(self, sum):
        result = 1 / (1 + np.exp(-sum))
        return result

    # Derivada da função sigmóide
    def calculate_derivative_sigmoid(self, sig_result):
        result = sig_result * (1 - sig_result)
        return result

    # Função de propagação
    def forward_propagation(self, entries):
        caches = []
        # 1.al=1 = x(i)
        a = entries

        for l in range(1, len(self.layers)):
            # 1.z(l=k) = θ(l=k-1) a(l=k-1)
            z = np.dot(self.weights['W' + str(l)], a) + self.weights['b' + str(l)]
            linear_cache = (a, self.weights['W' + str(l)], self.weights['b' + str(l)])
            a = self.calculate_sigmoid(z)
            cache = (linear_cache, z)
            caches.append(cache)

        return a, caches

    # Função para calcular o custo
    # Necessário inserir a regularização
    def calculate_cost(self, AL):
        cost = - np.sum(np.multiply(np.log(AL), self.outputs) + np.multiply((1 - self.outputs), np.log(1 - AL))) / self.num_examples
        return cost

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
    def backward_propagation(self, activation, caches):

        dAL = - (np.divide(self.outputs, activation) - np.divide(1 - self.outputs, 1 - activation))

        linear_cache, activation_cache = caches[len(caches) - 1]

        dZ = dAL * self.calculate_derivative_sigmoid(self.calculate_sigmoid(activation_cache))

        A_prev, W, b = linear_cache

        self.gradients["dW" + str(len(caches))] = (1 / self.num_examples) * np.dot(dZ, A_prev.T)
        self.gradients["db" + str(len(caches))] = (1 / self.num_examples) * np.sum(dZ, axis=1, keepdims=True)
        self.gradients["dA" + str(len(caches) - 1)] = np.dot(W.T, dZ)

        for l in reversed(range(len(caches) - 1)):
            linear_cache, activation_cache = caches[l]
            dZ = self.gradients["dA" + str(l + 1)] * self.calculate_derivative_sigmoid(self.calculate_sigmoid(activation_cache))

            A_prev, W, b = linear_cache
            m = A_prev.shape[1]

            self.gradients["dW" + str(l + 1)] = 1 / m * np.dot(dZ, A_prev.T)
            self.gradients["db" + str(l + 1)] = 1 / m * np.sum(dZ, axis=1, keepdims=True)
            self.gradients["dA" + str(l)] = np.dot(W.T, dZ)

    def update_parameters(self):
        for l in range(len(self.weights) // 2):
            self.weights["W" + str(l + 1)] -= self.learning_rate * self.gradients["dW" + str(l + 1)]
            self.weights["b" + str(l + 1)] -= self.learning_rate * self.gradients["db" + str(l + 1)]

    def train(self):

        np.random.seed(1)

        # Inicia pesos aleatórios
        self.init_weights()

        # Loop (gradient descent)
        for i in range(0, self.epochs):

            # Propagar o exemplo pela rede, calculando sua saída fθ(x)
            AL, caches = self.forward_propagation(self.entries)

            # Calcula custo
            cost = self.calculate_cost(AL)
            erroCamadaSaida = AL - self.outputs
            mediaAbsoluta = np.mean(np.abs(erroCamadaSaida))

            # Backward propagation.
            self.backward_propagation(AL, caches)

            # Atualiza parâmetros.
            self.update_parameters()

            # Print the cost every 100 training example
            if i % 1000 == 0:
                print("Cost after iteration %i: %f" % (i, cost))
                self.J.append(cost)
                print(AL)
                print("Erro: %.8f" % mediaAbsoluta)

        return self.weights

    def predict(self, entry):

        # Propagação
        result, caches = self.forward_propagation(entry)

        return result

    def accuracy(self, input, output):

        error=(self.predict(input)-output)
        accuracy=1-abs(error[0][0])

        print("Accuracy: " + str(accuracy))
        return accuracy







