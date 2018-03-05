import neuralNetwork as nn
import os
import numpy as np
import dill
import scipy.ndimage


BASE_FOLDER = os.path.abspath(os.path.dirname(__file__))

mnist_train_full = os.path.join(BASE_FOLDER, "resources/mnist_train.csv")

with open(mnist_train_full, "r") as f:
    data_train_list = f.readlines()

input_nodes = 784
hidden_nodes = 200
output_nodes = 10
learning_rate = 0.1

n = nn.NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

times = 5
for i in range(times):
    for record in data_train_list:
        all_values = record.split(',')
        lable = int(all_values[0])
        inputs = (np.asfarray(all_values[1:]) / 255 * 0.99) + 0.01
        targets = np.zeros(output_nodes) + 0.01
        targets[lable] = 0.99
        n.train(inputs, targets)

        ## create rotated variations
        # rotated anticlockwise by x degrees +10
        inputs_plusx_img = scipy.ndimage.interpolation.rotate(inputs.reshape(28, 28), 10, cval=0.01, order=1,
                                                              reshape=False)
        n.train(inputs_plusx_img.reshape(784), targets)
        # rotated clockwise by x degrees -10
        inputs_minusx_img = scipy.ndimage.interpolation.rotate(inputs.reshape(28, 28), -10, cval=0.01, order=1,
                                                               reshape=False)
        n.train(inputs_minusx_img.reshape(784), targets)

    print "finished ", i + 1

# save our trained nn
dill.dump(n,open(os.path.join(BASE_FOLDER, "resources/trained_nn.pickle"),'wb'))


