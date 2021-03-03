from layers import Linear_Layer, ReLU_Layer, Tanh_Layer, Sigmoid_Layer, Leaky_ReLU_Layer, Swish_Layer
import json
from pathlib import Path
import pickle
from auxiliary import save_as_pickle, load_pickle

class NeuralNetwork():
    """
    Basic Neural Network Class that combines multiple layers. Uses them, to
    implement the foward and backward passing.
    """
    def __init__(self, layers = None, load_saved_weights = False,
        model_file_path = None):
        """
        Initialization function.

        Args:
            layers (list):                  List of Layer objects.
            load_saved_weights (bool):      If set, rather than training, load
                                                weight of pretrained model
            model_file_path (str):          Path to saved pretrained weights
        """
        if not load_saved_weights and layers:
            # if weights are not to be loaded and layers where supplied, use
            # them
            self.layers = layers
        elif load_saved_weights and model_file_path:
            # if to be loaded and path was supplied, load them
            self.layers = self._load_layers(model_file_path)
        else:
            raise ValueError("When loading weights, supply path.")

    def forward(self, x):
        """
        Neural Network forward step.

        Args:
            x (nd.array):       Input to neural network
        Returns:
            x (nd.array):       Output of the neural network
        """
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def _backpropagation(self, error):
        """
        Backpropagate the error through the network. Note that during the
        backwards passing, the error is individually and internally saved by
        the layers for later updating.

        Args:
            error (nd.array):   Error of the loss function.
        """
        # backward passes
        for layer in reversed(self.layers):
            error = layer.backward(error)

    def _update_weights(self, lr):
        """
        Updating the weights, using the previously saved gradients within the
        layers themselves.

        Args:
            lr (float):         Learning Rate
        """
        # update layer weights
        for layer in self.layers:
            layer._update_weights(lr)

    def _save_model(self, model_path):
        """
        Function for saving the model architecture and weight as a dictionary
        in a pickle format.

        Args:
            model_path (str):   Path to folder where model is to be saved
                                (ending with a /)
        """
        # dictionary for saving model information
        dic = {}

        # collect all information in a dictionary
        for i, layer in enumerate(self.layers):
            dic[f"layer_{i+1}"] = {}
            dic[f"layer_{i+1}"]["type"] = layer.name
            dic[f"layer_{i+1}"]["weight_shape"] = layer.weights.shape
            #dic[f"layer_{i+1}"]["bias_shape"] = layer.bias.shape
            dic[f"layer_{i+1}"]["weights"] = layer.weights
            dic[f"layer_{i+1}"]["bias"] = layer.bias

        # if the folder is not yet created, do so
        Path(model_path).mkdir(exist_ok=True)

        # save the dictionary as a pickle
        save_as_pickle(dic, model_path + "model.pickle")


    def _load_layers(self, model_file_path):
        """
        Function for loading saved model architecture and weights and
        initializing this NeuralNetwork with it.

        Args:
            model_file_path (str): Path to the saved pickle file

        Returns:
            layers (list): List of initialized Layer Objects
        """
        # dictionary for translating layer types into actual objects
        layer_names = {"Linear": Linear_Layer,
                        "ReLU": ReLU_Layer,
                        "Tanh": Tanh_Layer,
                        "Sigmoid": Sigmoid_Layer,
                        "Leaky_ReLU": Leaky_ReLU_Layer,
                        "Swish": Swish_Layer}

        # loading the saved parameters
        dic = load_pickle(model_file_path)

        # list for storing the loaded layers
        layers = []

        # for each saved layer
        for i in range(len(dic)):
            # extract the information
            layer_type = dic[f"layer_{i+1}"]["type"]
            layer_weight_shape = dic[f"layer_{i+1}"]["weight_shape"]
            #layer_bias_shape = dic[f"layer_{i+1}"]["bias_shape"]
            layer_weights = dic[f"layer_{i+1}"]["weights"]
            layer_bias = dic[f"layer_{i+1}"]["bias"]

            # create a new layer of the same type and set it weights to the
            # saved ones
            loaded_layer = layer_names[layer_type](layer_weight_shape[1], layer_weight_shape[0])
            loaded_layer.weights = layer_weights
            loaded_layer.bias = layer_bias

            # add it to the layer list
            layers.append(loaded_layer)

        return layers
