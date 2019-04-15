
# coding: utf-8

# In[2]:

import convolutional_neural_network
import sys

diretorio = sys.argv[1:]
convolutional_neural_network.open_file("save_weights.hdf5")
convolutional_neural_network.test_image(diretorio[0])

