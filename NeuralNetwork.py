import numpy
from scipy.special import expit

class NeuralNetwork():
    def __init__(self, inputNodes, hiddenNodes, outputNodes, learningRate):
        print(f'Starting the Neural Network... Inputs: {inputNodes}, Hidden Nodes: {hiddenNodes}, Outputs: {outputNodes}')

        self.input_nodes = inputNodes
        self.hidden_nodes = hiddenNodes
        self.output_nodes = outputNodes
        self.learning_rate = learningRate
        #self.activation_func = lambda x: 1 / ( 1 + numpy.exp( -x ) )
        self.activation_func = lambda x: expit(x)

        self.weights_ih = (numpy.random.rand(self.hidden_nodes, self.input_nodes) - 0.5)
        self.weights_ho = (numpy.random.rand(self.output_nodes, self.hidden_nodes) - 0.5)

    def train(self, inputList, targetList):
        inputs = numpy.array(inputList, ndmin=2).T
        targets = numpy.array(targetList, ndmin=2).T

        hiddenInputs = numpy.dot(self.weights_ih, inputs)
        hiddenOutputs = self.activation_func(hiddenInputs)

        finalInputs = numpy.dot(self.weights_ho, hiddenOutputs)
        finalOutputs = self.activation_func(finalInputs)

        outputErrors = targets - finalOutputs
        

        hiddenErrors = numpy.dot(self.weights_ho.T, outputErrors)
        print(outputErrors)

        self.weights_ho += self.learning_rate * numpy.dot((outputErrors * finalOutputs * (1.0 - finalOutputs)), numpy.transpose(hiddenOutputs))
        self.weights_ih += self.learning_rate * numpy.dot((hiddenErrors * hiddenOutputs * (1.0 - hiddenOutputs)), numpy.transpose(inputs))

    def query(self, inputList):
        inputs = numpy.array(inputList, ndmin=2).T

        hiddenInputs = numpy.dot(self.weights_ih, inputs)
        hiddenOutputs = self.activation_func(hiddenInputs)

        finalInputs = numpy.dot(self.weights_ho, hiddenOutputs)
        finalOutputs = self.activation_func(finalInputs)

        return finalOutputs

