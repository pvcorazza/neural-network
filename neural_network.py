import copy

import numpy as np

class NeuralNetwork:
    def __init__(self, reg_factor, entries, outputs, layers, learning_rate=1.2, iterations = 1000000, weights=None):

        # Definição dos parâmetros
        self.reg_factor = reg_factor
        self.entries = entries
        self.outputs = outputs
        self.layers = layers
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.num_examples = outputs.shape[0]
        self.weights = weights
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

        return self.weights

    # Função sigmóide
    def calculate_sigmoid(self, sum):
        result = 1 / (1 + np.exp(-sum))
        return result

    # Derivada da função sigmóide
    def calculate_derivative_sigmoid(self, sig_result):
        result = sig_result * (1 - sig_result)
        return result

    # Função de propagação
    def forward_propagation(self, entries, weights):

        self.activations = {}
        # 1.al=1 = x(i)
        a = entries
        self.activations['a1'] = a

        for l in range(1, len(self.layers)):
            # 1.z(l=k) = θ(l=k-1) a(l=k-1)
            z = np.dot(weights['W' + str(l)], self.activations['a'+str(l)]) + weights['b' + str(l)]
            # linear_cache = (a, weights['W' + str(l)], weights['b' + str(l)])
            a = self.calculate_sigmoid(z)
            self.activations['a' + str(l+1)] = a

        return a

    # Função para calcular o custo
    # Necessário inserir a regularização
    def calculate_cost(self, AL):

        cost = - np.sum(np.multiply(np.log(AL), self.outputs) + np.multiply((1 - self.outputs), np.log(1 - AL))) / self.num_examples
        regularization = self.calculate_regularization()

        return cost + regularization

    def calculate_regularization(self):

        sum = 0

        for i in range(len(self.weights) // 2):
            sum += np.sum(np.square(self.weights["W" + str(i + 1)]))

        regularization = (1 / self.num_examples) * (self.reg_factor / 2) * sum

        return regularization

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
    def backward_propagation(self):

        self.num_examples = self.entries.shape[1]

        # dAL = - (np.divide(self.outputs, activation) - np.divide(1 - self.outputs, 1 - activation))

        # linear_cache, activation_cache = caches[len(caches) - 1]

        dZ = self.activations["a" + str(len(self.activations))] - self.outputs

        self.gradients["dW" + str(len(self.activations)-1)] = (1 / self.num_examples) * np.dot(dZ, self.activations["a" + str(len(self.activations)-1)].T) + self.reg_factor/self.num_examples * self.weights["W" + str(len(self.activations)-1)]
        self.gradients["db" + str(len(self.activations)-1)] = (1 / self.num_examples) * np.sum(dZ, axis=1, keepdims=True)

        for l in reversed(range(2, len(self.activations))):

            dZ = np.dot(self.weights["W" + str(l)].T, dZ) * self.calculate_derivative_sigmoid(self.activations["a" + str(l)])

            self.gradients["dW" + str(l-1)] = (1 / self.num_examples) * np.dot(dZ, self.activations["a" + str(l-1)].T) + self.reg_factor/self.num_examples * self.weights["W" + str(l - 1)]
            self.gradients["db" + str(l-1)] = (1 / self.num_examples) * np.sum(dZ, axis=1, keepdims=True)

        return self.gradients

    def update_weights(self):
        for l in range(len(self.weights) // 2):
            self.weights["W" + str(l + 1)] -= self.learning_rate * self.gradients["dW" + str(l + 1)]
            self.weights["b" + str(l + 1)] -= self.learning_rate * self.gradients["db" + str(l + 1)]

    def train(self):

        # np.random.seed(1)

        # Inicia pesos aleatórios
        self.init_weights()

        # Loop (gradient descent)
        for i in range(0, self.iterations):

            # Propagar o exemplo pela rede, calculando sua saída fθ(x)
            AL = self.forward_propagation(self.entries, self.weights)

            # Calcula custo
            cost = self.calculate_cost(AL)

            # Backward propagation.
            self.backward_propagation()

            # Copia pesos e gradientes para verificação, antes da atualização dos pesos
            weights = copy.deepcopy(self.weights)
            gradients = copy.deepcopy(self.gradients)

            # grads, approx = self.gradient_check_n(self.weights, self.gradients)
            # self.compare_gradients(grads, approx)

            # Atualiza os pesos
            self.update_weights()

            # Print the cost every 100 training example
            if i % 1000 == 0:
                print("Cost after iteration %i: %f" % (i, cost))
                self.J.append(cost)
                # print(AL)
                print("Erro: %.8f" % mediaAbsoluta)

        return weights, gradients

    def predict(self, entry):

        # Propagação
        result, caches = self.forward_propagation(entry)

        return result

    def accuracy(self, input, output):

        error=(self.predict(input)-output)
        accuracy=1-abs(error[0][0])

        print("Accuracy: " + str(accuracy))
        return accuracy

    def predict2(self, X, y):
        """
        This function is used to predict the results of a  n-layer neural network.

        Arguments:
        X -- data set of examples you would like to label
        parameters -- parameters of the trained model

        Returns:
        p -- predictions for the given dataset X
        """

        m = X.shape[1]
        p = np.zeros((1, m), dtype=np.int)

        # Forward propagation
        a3 = self.forward_propagation(X, self.weights)

        # convert probas to 0/1 predictions
        for i in range(0, a3.shape[1]):
            if a3[0, i] > 0.5:
                p[0, i] = 1
            else:
                p[0, i] = 0

        # print results

        # print ("predictions: " + str(p[0,:]))
        # print ("true labels: " + str(y[0,:]))
        print("Accuracy: " + str(np.mean((p[0, :] == y[0, :]))))

        return p

    def weights_to_vector(self, weights):

        vector_weights = np.matrix([]).reshape((0,1))

        for i in range(len(self.weights) // 2):

            vector_w = np.reshape(weights["W" + str(i + 1)], (-1, 1))
            vector_b = np.reshape(weights["b" + str(i + 1)], (-1, 1))
            concat = np.concatenate((vector_w, vector_b),axis=0)
            vector_weights = np.concatenate((vector_weights, concat),axis=0)

        return vector_weights


    def gradients_to_vector(self, gradients):

        vector_gradients = np.matrix([]).reshape((0,1))
        for i in range(len(self.weights) // 2):

            vector_w = np.reshape(gradients["dW" + str(i + 1)], (-1, 1))
            vector_b = np.reshape(gradients["db" + str(i + 1)], (-1, 1))
            concat = np.concatenate((vector_w, vector_b),axis=0)
            vector_gradients = np.concatenate((vector_gradients, concat),axis=0)

        return vector_gradients


    def vector_to_dictionary(self, vector_weights):

        dic = {}

        for l in range(1, len(self.layers)):

            final_weight = (self.layers[l]*self.layers[l - 1])
            final_bias = (self.layers[l])

            dic['W' + str(l)] = vector_weights[0:self.layers[l]*self.layers[l - 1]].reshape((self.layers[l], self.layers[l - 1]))
            vector_weights = np.delete(vector_weights, np.s_[0:final_weight], axis=0)

            dic['b' + str(l)] = vector_weights[0:self.layers[l]].reshape((self.layers[l], 1))
            vector_weights = np.delete(vector_weights, np.s_[0:final_bias], axis=0)

        return dic

    def gradient_check_n(self, weights, gradients, epsilon=1e-7):

        # Descarta regularização
        for i in range(1, len(gradients)-1):
            gradients["dW" + str(i)] = gradients["dW" + str(i)] - self.reg_factor / self.num_examples * self.weights["W" + str(i)]

        vector_weights = self.weights_to_vector(weights)
        vector_gradients = self.gradients_to_vector(gradients)

        num_parameters = vector_weights.shape[0]
        J_plus = np.zeros((num_parameters, 1))
        J_minus = np.zeros((num_parameters, 1))
        grad_approx = np.zeros((num_parameters, 1))

        for i in range(num_parameters):

            thetaplus = copy.deepcopy(vector_weights)
            thetaplus[i][0] += epsilon
            a = self.forward_propagation(self.entries, self.vector_to_dictionary(thetaplus))
            J_plus[i] = self.calculate_cost(a) - self.calculate_regularization()

            thetaminus = np.copy(vector_weights)
            thetaminus[i][0] -= epsilon
            a = self.forward_propagation(self.entries, self.vector_to_dictionary(thetaminus))
            J_minus[i] = self.calculate_cost(a) - self.calculate_regularization()

            grad_approx[i] = (J_plus[i] - J_minus[i]) / (2 * epsilon)

        return vector_gradients, grad_approx

    def compare_gradients(self, backward_gradients, grad_approx):
        numerator = np.linalg.norm(backward_gradients - grad_approx)
        denominator = np.linalg.norm(backward_gradients) + np.linalg.norm(grad_approx)
        difference = numerator / denominator  # Step 3'

        if difference > 2e-6:
            print(
                "\033[93m" + "There is a mistake in the backward propagation! difference = " + str(difference) + "\033[0m")
        else:
            print(
                "\033[92m" + "Your backward propagation works perfectly fine! difference = " + str(difference) + "\033[0m")

        return difference

    def get_gradients_to_compare(self, entries, weights=None):
        if (weights==None):
            weights = self.init_weights()

        # Propagar o exemplo pela rede, calculando sua saída fθ(x)
        a = self.forward_propagation(entries, weights)
        print("Saida preditas para os exemplos")
        print(a)
        print("Custo do dataset")
        # Calcula custo
        cost = self.calculate_cost(a)
        print(cost)
        # Backward propagation.
        gradients = self.backward_propagation()

        grads, approx = self.gradient_check_n(weights, gradients)
        print("Gradientes do BackPropagation")
        print(grads)
        print("Gradientes da Aproximação")
        print(approx)
        exit()

        return grads, approx