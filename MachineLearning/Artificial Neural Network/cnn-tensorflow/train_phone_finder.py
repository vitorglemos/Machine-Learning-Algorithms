
# coding: utf-8

# In[1]:

import convolutional_neural_network

train_data = convolutional_neural_network.preparar_arquivos_1()
X_train, Y_train = convolutional_neural_network.preparar_arquivos_2(train_data)
print("************************************************")
convolutional_neural_network.train_Neural_Network(X_train,Y_train)

