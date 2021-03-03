from tkinter import *

from test import training

def deep_learning_form():

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

        # TODO check if lengths of lists match
        if True:

            # close root window
            root.quit()

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

    root.mainloop()

if __name__=="__main__":
    deep_learning_form()
