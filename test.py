from data import generate_standard_data
from networks import NeuralNetwork
from layers import Linear_Layer, ReLU_Layer, Tanh_Layer, Sigmoid_Layer, Leaky_ReLU_Layer, Swish_Layer
from losses import LossMSE
from auxiliary import calculate_performance, plot_data, plot_performance, early_stopping

import numpy as np


# TODO use torch.empty instead of numpy.array
# TODO comments
# TODO check input for wrong things

np.random.seed(0)

MODEL_PATH = "./models/"

def training(num_inputs = 2, num_outputs = 1, nr_hidden = 3,
            num_hidden = [25, 25, 25], activation_functions = [2, 6, 3],
            learning_rate = 0.0001, training_epochs = 500,
            stopping_condition_n = 3, verbose = False):

    print("[!] STARTING TRAINING")

    # translation dictionary for activation_functions
    number_to_activation = [Linear_Layer, ReLU_Layer, Tanh_Layer,
                        Sigmoid_Layer, Leaky_ReLU_Layer, Swish_Layer]

    # determine the layers to be used
    used_layers = [number_to_activation[i-1] for i in activation_functions]

    # merging input + output + hidden neuron numbers into one structure
    used_neurons = [num_inputs] + num_hidden + [num_outputs]

    # initialized layers
    initialized_layers = [layer_class(used_neurons[i], used_neurons[i+1]) for i, layer_class in enumerate(used_layers)]
    # add initialized output layer
    initialized_layers += [Linear_Layer(used_neurons[-2], used_neurons[-1])]

    # building the model
    network = NeuralNetwork(layers = initialized_layers)

    # generate the test data
    X_train, Y_train, X_test, Y_test = generate_standard_data()

    # initialize loss function
    MSE = LossMSE()


    # lists for collecting the MSE erros and the accuracies
    training_mse = []
    training_accuracies = []
    testing_mse = []
    testing_accuracies = []

    for epoch in range(training_epochs):

        # same as above, but only for this epoch
        epoch_mse = []
        epoch_accuracies = []


        # go through all samples in this epoch
        for x, t in zip(X_train, Y_train):
            ##### TRAINING
            # calculate the networks output
            out = network.forward(x.reshape((2,1)))

            # calculate whether the prediction was correct not
            predicted_label = 1 if out >= 0.5 else 0
            correct = int(t == predicted_label)

            epoch_accuracies.append(correct)

            # calculate MSE of this
            mse_error = MSE(t, out)

            epoch_mse.append(mse_error)

            # calculate error for backpropagation
            error = MSE.backward()

            # backpropagate the error
            network._backpropagation(error, learning_rate)

        # compute the mean accuracy and mse for this epoch and store it
        training_mse.append((epoch, np.mean(epoch_mse)))
        training_accuracies.append((epoch, np.mean(epoch_accuracies)))

        if verbose and epoch % 10 == 0:
            print(f"[*TRAINING*] EPOCH {epoch:4} |   MSE: {training_mse[-1][1]:2.3f} | Accuracy: {training_accuracies[-1][1]:1.3f}")

        ##### INTERMEDIATE TESTING
        # calculate mse and accuracy on the test data with the current model
        mse_error, accuracy, predictions = calculate_performance(X_test, Y_test, network)
        # save them in the corresponding lists for later plotting
        testing_mse.append((epoch, mse_error))
        testing_accuracies.append((epoch, accuracy))
        #print(f"[*TESTING**] EPOCH {epoch:4} |   MSE: {testing_mse[-1][1]:2.3f} | Accuracy: {testing_accuracies[-1][1]:1.3f}")

        ##### CHECK FOR EARLY STOPPING
        if early_stopping(testing_mse, n = stopping_condition_n):
            break


    # save the current model
    network._save_model(MODEL_PATH)

    # plot training/test error/accuracy
    plot_performance(training_mse, training_accuracies, testing_mse, testing_accuracies)

    print("[+] TRAINING FINISHED")

    # return the network

def testing(network = None):
    print("[!] STARTING TEST")
    # generate the test data
    _, _, X_test, Y_test = generate_standard_data()

    # check if a network was supplied
    if not network:
        # if not load network from saved model file
        network = NeuralNetwork(load_saved_weights = True, model_file_path = MODEL_PATH + "model.pickle")

    mse_error, accuracy, predictions = calculate_performance(X_test, Y_test, network)

    print(f"[+] TEST FINISHED WITH ACCURACY OF {accuracy:.2f}% AND MSE OF {mse_error:.2f}")


if __name__=="__main__":
    trained_network = training()
    testing(trained_network)
