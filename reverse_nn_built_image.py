import os
import numpy as np
import matplotlib.pyplot as plt
import dill

BASE_FOLDER = os.path.abspath(os.path.dirname(__file__))

mnist_test_full = os.path.join(BASE_FOLDER, "resources/mnist_test.csv")
nn_trained_path = os.path.join(BASE_FOLDER, "resources/trained_nn.pickle")

with open(mnist_test_full, "r") as f:
    data_test_list = f.readlines()

with open(nn_trained_path, "rb") as inp:
    n = dill.load(inp)

correct_answers = 0.0
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

lable = 4
targets = np.zeros(n.onodes) + 0.01
targets[lable] = 0.99
outputs = n.backquery(targets)
print outputs

plt.imshow(outputs.reshape(28,28), cmap='Greys', interpolation='None')
plt.show()
