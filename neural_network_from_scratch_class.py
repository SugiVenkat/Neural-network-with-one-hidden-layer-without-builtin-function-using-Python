import numpy as np
# setting activation functions outside class, since they are static
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_der(x):
    return x * (1 - x)

class NeuralNetwork:
    def __init__(self, inputs, outputs, no_of_neurons_hidden = 3):
        self.inputs = inputs
        self.outputs = outputs
        self.no_of_neurons_hidden = no_of_neurons_hidden
        self.no_of_neurons_input = self.inputs.shape[1]
        self.no_of_neurons_output = self.outputs.shape[1]

    def weights_biases_initialize(self):
        # initializing weights, biases for hidden and output layers - Wh1, bh1, Wo1, bo1
        self.Wh1 = np.random.rand(self.no_of_neurons_input, self.no_of_neurons_hidden)
        self.bh1 = np.random.rand(1, self.no_of_neurons_hidden)
        self.Wo = np.random.rand(self.no_of_neurons_hidden, self.no_of_neurons_output)
        self.bo = np.random.rand(1, self.no_of_neurons_output)

    def forward_prop(self, input_fp):
        # forward prop results of hidden layer 1 and output layer computations
        self.hidden1_res = sigmoid(np.dot(input_fp, self.Wh1) + self.bh1)
        self.output_res = sigmoid(np.dot(self.hidden1_res, self.Wo) + self.bo)

    def back_prop(self):
        error = self.outputs - self.output_res
        diff_Wo = np.dot(self.hidden1_res.T, (error * sigmoid_der(self.output_res)))
        diff_bo = np.sum(error * sigmoid_der(self.output_res), axis=0, keepdims=True)
        diff_Wh1 = np.dot(inputs.T, np.dot(error * sigmoid_der(self.output_res), self.Wo.T) * sigmoid_der(self.hidden1_res))
        diff_bh1 = np.sum(np.dot(error * sigmoid_der(self.output_res), self.Wo.T) * sigmoid_der(self.hidden1_res), axis=0,
                          keepdims=True)
        self.Wh1 += diff_Wh1
        self.bh1 += diff_bh1
        self.Wo += diff_Wo
        self.bo += diff_bo

    def train_nn(self, number_of_iterations=5000):
        self.number_of_iterations = number_of_iterations
        self.weights_biases_initialize()
        for i in range(self.number_of_iterations):
            self.forward_prop(self.inputs)
            self.back_prop()
        print(self.output_res)

    def test_nn(self, inputs_test):
        self.forward_prop(inputs_test)
        print(self.output_res)


inputs = np.array([[0, 1, 0], [1, 1, 0], [1, 0, 1], [0, 0, 1], [1, 0, 0]])
outputs = np.array([[0, 1, 1], [1, 0, 0], [1, 1, 1], [0, 1, 1], [0, 0, 0]])
neural1 = NeuralNetwork(inputs,outputs, 5)
neural1.train_nn(6000)
inputs_test = [1, 1, 0]
neural1.test_nn(inputs_test)











