from data import generate_standard_data, plot_data
from networks import NeuralNetwork
from layers import Linear_Layer, ReLU_Layer, Tanh_Layer, Sigmoid_Layer, Leaky_ReLU_Layer, Swish_Layer
from losses import LossMSE

import numpy as np
X, Y, X1, Y1, = generate_standard_data()
#plot_data(X1,Y1)

def test_accuracy(X_test, Y_test):
    prediction = network.forward(X1.T)[0]
    prediction[prediction>=0.5] = 1
    prediction[prediction<0.5] = 0
    prediction = prediction.astype(int)
    accuracy = np.mean(prediction == Y_test)
    plot_data(X_test,prediction)
    return accuracy, prediction

# Seed random generator to get consistent results
np.random.seed(0)
# Build the model
num_inputs = 2
num_hidden = 25
num_output = 1
network = NeuralNetwork([
    Swish_Layer(num_inputs, num_hidden),
    Swish_Layer(num_hidden, num_hidden),
    Swish_Layer(num_hidden, num_hidden),
    Linear_Layer(num_hidden, num_output)
  ])

print("Training:")
learning_rate = 0.0001
MSE = LossMSE()



for epoch in range(1500):
  errors = []
  correct = []
  for x, target in zip(X, Y):
    #print(x)
    # Forward
    #print("\n"*10)
    #print(x)
    output = network.forward(x.reshape((2,1)))
    #print(output)
    correct.append(int(target == round(output[0,0])))
    #print(output)
    # Compute the error
    mse = MSE(target, output)
    error = MSE.backward()
    errors.append(error)

    # Back-propagate the error
    network._backpropagation(error, learning_rate)

  # Compute the Mean Squared Error of all examples each 100 epoch
  if epoch % 20 == 0:
    print("  Epoch %3d   MSE: %.3f" % (epoch, mse))
    print(np.mean(correct))
  if epoch % 200 == 0:
    print(test_accuracy(X1, Y1)[0])

acc, pred = test_accuracy(X1, Y1)
