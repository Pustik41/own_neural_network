import neuralNetwork as nn
import os
import numpy as np
import matplotlib.pyplot as plt

BASE_FOLDER = os.path.abspath(os.path.dirname(__file__))

mnist_test_10 = os.path.join(BASE_FOLDER, "resources/mnist_test_10.csv")
mnist_test_full = os.path.join(BASE_FOLDER, "resources/mnist_test.csv")
mnist_train_100 = os.path.join(BASE_FOLDER, "resources/mnist_train_100.csv")
mnist_train_full = os.path.join(BASE_FOLDER, "resources/mnist_train.csv")

with open(mnist_train_100, "r") as f:
    data_train_list = f.readlines()

with open(mnist_test_10, "r") as f:
    data_test_list = f.readlines()

#first_number_data = data_train_list[0].split(',')
#image_first_number = np.asfarray(first_number_data[1:]).reshape((28,28))
#plt.imshow(image_first_number, cmap="Greys", interpolation=None)
#plt.show()
#all_values = first_number_data = data_train_list[0].split(',')
## convert values 0-255 to diapasone from 0.01-1.0 for inputs signal
#scaled_input = ((np.asfarray(all_values[1:]) / 255) * 0.99) + 0.01
#
#onodes = 10
#targets = np.zeros(onodes) + 0.01
#targets[int(all_values[0])]  = 0.99

input_nodes = 784
hidden_nodes = 100
output_nodes = 10
learning_rate = 0.1

n = nn.NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

# train neural network
times = 5
for i in range(times):
    for record in data_train_list:
        all_values = record.split(',')
        lable = int(all_values[0])
        inputs = (np.asfarray(all_values[1:]) / 255 * 0.99) + 0.01
        targets = np.zeros(output_nodes) + 0.01
        targets[lable] = 0.99
        n.train(inputs, targets)

correct_answers = 0.0

# testing neural networ
for record in data_test_list:
    all_values = record.split(',')
    correct_lable = int(all_values[0])
    inputs = ((np.asfarray(all_values[1:]) / 255) * 0.99) + 0.01
    outputs = n.query(inputs)
    lable = np.argmax(outputs)

    if lable == correct_lable:
        correct_answers += 1.0

accuracy  = correct_answers / len(data_test_list)
print "Accuracy = ", accuracy















