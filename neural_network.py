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
        self.epochs = 1000000
        self.learning_rate = 1.2
        self.momentum = 1
        self.J = 0
        self.num_examples = np.shape(self.outputs)[1]

    # Define pesos aleatórios para a rede
    def init_random_weights(self):

        # Camada de entrada para camada oculta
        self.W1 = 2*np.random.randn(self.num_hidden, self.num_inputs) - 1
        # Bias da camada de entrada
        self.b1 = np.ones(shape=(self.num_hidden, 1), dtype=int)
        # Camada oculta para camada de saída
        self.W2 = 2*np.random.randn(self.num_outputs, self.num_hidden) - 1
        # Bias da camada oculta
        self.b2 = np.ones(shape=(self.num_outputs, 1), dtype=int)

    # Função sigmóide
    def calculate_sigmoid(self, sum):
        result = 1 / (1 + np.exp(-sum))
        return result

    # Derivada da função sigmóide
    def calculate_derivative_sigmoid(self, sig_result):
        result = sig_result * (1-sig_result)
        return result

    # Calcula novos pesos para camada
    def calculate_new_weights(self, weights, layer, delta):
        pesosNovo0 = np.dot(delta, layer.T)
        weights = (weights * self.momentum) + ( pesosNovo0 * self.learning_rate)
        return weights

    def calculate_cost(self):
        cost = - np.sum(np.multiply(np.log(self.a3), self.outputs) + np.multiply((1 - self.outputs), np.log(1 - self.a3))) / self.num_examples
        return cost

    # Função de propagação
    def forward_propagation(self, entries):

        # 1.al=1 = x(i)
        self.a1 = entries

        # 1.z(l=k) = θ(l=k-1) a(l=k-1)
        z2 = np.dot(self.W1, self.a1) + self.b1

        # 2.a(l=k) = g(z(l=k))
        self.a2 = self.calculate_sigmoid(z2)

        # 4.z(l=L) = θ(l=L-1) a(l=L-1)
        z3 = np.dot(self.W2, self.a2) + self.b2
        # 5. a(l=L) = g(z(l=L))
        self.a3 = self.calculate_sigmoid(z3)
        # 6.Retorna fθ(x(i)) = a(l=L)
        return self.a3

    # Função de backpropagation
    def back_propagation(self, a3):
        # Calcular o valor de 𝛿 para os neurônios da camada de saída:
        # 1.2.𝛿(l=L) = fθ(x(i)) - y(i)
        error = a3 - self.outputs

        #1.3.Para cada camada k=L-1…2 // calcula os deltas para as camadas ocultas
        # 𝛿(l=k) = [θ(l=k)]T 𝛿(l=k+1) .* a(l=k) .* (1-a(l=k))

        delta2 = error * self.calculate_derivative_sigmoid(a3)

        delta1 = np.dot(self.W2.T, delta2) * self.calculate_derivative_sigmoid(self.a2)

        return delta2, delta1

    def train(self):
        self.init_random_weights()
        for i in range (self.epochs):

            #Propagar o exemplo pela rede, calculando sua saída fθ(x)
            a3 = self.forward_propagation(self.entries)

            # #Calcula custo
            # cost = self.calculate_cost()

            erroCamadaSaida = self.a3 - self.outputs
            mediaAbsoluta = np.mean(np.abs(erroCamadaSaida))


            #Backpropagation
            output_delta, hidden_delta = self.back_propagation(a3)

            #Atualiza pesos
            self.W2 = self.calculate_new_weights(self.W2, self.a2, output_delta)
            self.W1 = self.calculate_new_weights(self.W1, self.entries, hidden_delta)

            # Print the cost every 1000 iterations
            if i % 1000 == 0:
                # print("Cost after iteration %i: %f" % (i, cost))
                print("Erro: " + str(mediaAbsoluta))




