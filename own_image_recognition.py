import neuralNetwork as nn
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc
import glob
import dill

BASE_FOLDER = os.path.abspath(os.path.dirname(__file__))
nn_trained_path = os.path.join(BASE_FOLDER, "resources/trained_nn.pickle")

with open(nn_trained_path, "rb") as inp:
    n = dill.load(inp)

img_data_set = []

def create_img_data_set():
    for img in glob.glob(os.path.join(BASE_FOLDER, "resources/?.png")):
        img_array = scipy.misc.imread(img, flatten=True)
        img_data = 255.0 - img_array.reshape(784)
        img_data = (img_data / 255.0 * 0.99) + 0.01
        lable = int(img[-5:-4])
        record = np.append(lable, img_data)
        img_data_set.append(record)


create_img_data_set()
print "dataset done"
#print type(img_data_set[0][0])

# test first img value five
# plt.imshow(img_data_set[3][1:].reshape((28,28)), cmap='Greys', interpolation='None')
# plt.show()

count_correct_answers = 0.0

for img in img_data_set:
    correct_label = img[0]
    inputs = img[1:]
    outputs = n.query(inputs)
    label = np.argmax(outputs)
    print("network says ", label)
    print "correct", correct_label
    if (label == correct_label):
        count_correct_answers += 1.0
        print ("match!")
    else:
        print (outputs)
        print ("no match!")

print "Accuracy = ", count_correct_answers / len(img_data_set)



