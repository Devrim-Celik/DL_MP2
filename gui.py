from tkinter import *

from test import training

def input_type_checking(nr_hidden_layers, nr_neurons_per_hidden,
                activation_functions, learning_rate, number_training_epochs):
    """
    For all variables defined by user input, check if the values are correct.

    Args:
        nr_hidden_layers (int):         User input for number of hidden layers
        nr_neurons_per_hidden (list):   User input for number neurons per layer
        activation_functions (list):    User input for activation functions per
                                            hidden layer
        learning_rate (float):          User input for learning rate
        number_training_epochs (int):   User input for number of training epochs
    """
    if not (nr_hidden_layers == len(nr_neurons_per_hidden)):
        raise ValueError(f"Neuron Number per Layer ({len(nr_neurons_per_hidden)}) have different Number of Elements then the Number of Hidden layers ({nr_hidden_layers}).")
    if not (min(nr_neurons_per_hidden) > 0):
        raise ValueError(f"Neuron numbers have to be bigger than 0, not {min(nr_neurons_per_hidden)}.")
    if not (nr_hidden_layers == len(activation_functions)):
        raise ValueError(f"Activation Function per Layer ({len(nr_neurons_per_hidden)}) have different Number of Elements then the Number of Hidden layers ({nr_hidden_layers}).")
    if not (max(activation_functions) <= 6 and min(activation_functions) >= 1):
        problem_number = max(activation_functions) if max(activation_functions) > 6 else min(activation_functions)
        raise ValueError(f"Activation numbers are encoded as numbers (1,2,3,4,5,6), not {problem_number}.")
    if not (learning_rate > 0):
        raise ValueError(f"Learning Rate has to be bigger than 0, not {learning_rate}.")
    if  not (number_training_epochs > 0):
        raise ValueError(f"Number of Training Epochs has to be bigger than 0, not {number_training_epochs}.")

def deep_learning_form():
    """
    GUI for entering parameters for deep learning training.
    """

    def submit():
        """
        Callback function for the Run Button. Extracts entered valus and
        calls deep learning framework test.
        """
        # extracts entered values from entry widgets
        nr_hidden_layers = int(nr_hidden_layers_entry.get())
        nr_neurons_per_hidden = [int(x) for x in list(nr_neurons_per_hidden_entry.get().split(","))]
        activation_functions = [int(x) for x in list(activation_functions_entry.get().split(","))]
        learning_rate = float(learning_rate_entry.get())
        number_training_epochs = int(number_training_epochs_entry.get())

        # check for user errors
        input_type_checking(nr_hidden_layers, nr_neurons_per_hidden,
                        activation_functions, learning_rate, number_training_epochs)

        # TODO check if lengths of lists match
        if True:

            # close root window
            root.destroy()

            # call test with given parameters
            training(2, 1, nr_hidden_layers, nr_neurons_per_hidden,
                            activation_functions, learning_rate,
                            number_training_epochs)

    # define root window
    root = Tk()
    root.geometry("1000x500")
    root.title("Deep Learning Setup")

    # create the heading widget
    heading = Label(text = "Deep Learning Setup", bg = "grey", fg = "black", width = "500", height = "3")
    heading.pack()

    # create text widgets
    nr_hidden_layers_text = Label(root, text="Number of Hidden Layers:")
    nr_neurons_per_hidden_text = Label(root, text="List of Number of Neurons per Hidden Layer:")
    activation_functions_text = Label(root, text="List of Activation Functions per Hidden Layer (with Linear = 1, ReLU = 2, Tanh = 3, Sigmoid = 4, Leaky ReLU = 5, Swish = 6):")
    learning_rate_text = Label(root, text="Learning Rate:")
    number_training_epochs_text = Label(root, text="Number of Training Epochs:")
    # place text widgets
    nr_hidden_layers_text.place(x = 15, y = 70)
    nr_neurons_per_hidden_text.place(x = 15, y = 140)
    activation_functions_text.place(x = 15, y = 210)
    learning_rate_text.place(x = 15, y = 280)
    number_training_epochs_text.place(x = 15, y = 350)

    # create entry widgets
    nr_hidden_layers_entry = Entry(root, width = "120")
    nr_neurons_per_hidden_entry = Entry(root, width = "120")
    activation_functions_entry = Entry(root, width = "120")
    learning_rate_entry = Entry(root, width = "120")
    number_training_epochs_entry = Entry(root, width = "120")
    # enter default values in entry widgets
    nr_hidden_layers_entry.insert(0, "3")
    nr_neurons_per_hidden_entry.insert(0, "25, 25, 25")
    activation_functions_entry.insert(0, "2, 6, 3")
    learning_rate_entry.insert(0, "0.0001")
    number_training_epochs_entry.insert(0, "500")
    # place entry widgets
    nr_hidden_layers_entry.place(x = 15, y = 100)
    nr_neurons_per_hidden_entry.place(x = 15, y = 170)
    activation_functions_entry.place(x = 15, y = 240)
    learning_rate_entry.place(x = 15, y = 310)
    number_training_epochs_entry.place(x = 15, y = 380)

    # create and place button widget
    run_button = Button(root, text = 'RUN', command = submit)
    run_button.place(x = 480, y = 430)

    # run GUI
    root.mainloop()

if __name__=="__main__":
    deep_learning_form()
