import numpy as np
import scipy.special

class NeuralNetwork:

    def __init__(self, inputnodes, hiddennodes, outputnodes, learninggrate):

        # number of nodes for each layer
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        # learning factor
        self.lr = learninggrate

        # weight coeficients whi -> between input and hidden , who -> between hidden and output
        self.wih = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))

        #sigmoid
        self.activation_function = lambda x: scipy.special.expit(x)
        #revers
        self.inverse_activation_function = lambda x: scipy.special.logit(x)

    def train(self, input_list, target_list):
        targets = np.array(target_list, ndmin=2).T
        inputs = np.array(input_list, ndmin=2).T

        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        # error = target - final_output
        output_errors = targets - final_outputs

        #calculate hidden errors
        hidden_errors = np.dot(self.who.T, output_errors)

        #update weights between hidden and output layers
        self.who += self.lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)),
                                     np.transpose(hidden_outputs))

        #update weights between input and hidden layers
        self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), np.transpose(inputs))

    def query(self, inputs_list):
        inputs = np.array(inputs_list, ndmin=2).T

        #inputs signals for hidden layer
        hidden_inputs = np.dot(self.wih, inputs)

        # output from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        #inputs to final layer
        final_inputs = np.dot(self.who, hidden_outputs)

        #output from final layer
        final_outputs = self.activation_function(final_inputs)

        return final_outputs

    # backquery the neural network
    # we'll use the same termnimology to each item,
    # eg target are the values at the right of the network, albeit used as input
    # eg hidden_output is the signal to the right of the middle nodes
    def backquery(self, targets_list):
        # transpose the targets list to a vertical array
        final_outputs = np.array(targets_list, ndmin=2).T
        # calculate the signal into the final output layer
        final_inputs = self.inverse_activation_function(final_outputs)
        # calculate the signal out of the hidden layer
        hidden_outputs = np.dot(self.who.T, final_inputs)
        # scale them back to 0.01 to .99
        hidden_outputs -= np.min(hidden_outputs)
        hidden_outputs /= np.max(hidden_outputs)
        hidden_outputs *= 0.98
        hidden_outputs += 0.01
        # calculate the signal into the hidden layer
        hidden_inputs = self.inverse_activation_function(hidden_outputs)
        # calculate the signal out of the input layer
        inputs = np.dot(self.wih.T, hidden_inputs)
        # scale them back to 0.01 to .99
        inputs -= np.min(inputs)
        inputs /= np.max(inputs)
        inputs *= 0.98
        inputs += 0.01
        return inputs


