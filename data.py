import numpy as np
import matplotlib.pyplot as plt

def generate_standard_data(N=1000, center_value=0.5, radius=1/np.sqrt(2*np.pi)):
    """
    Function for generating training and test data, where points are generated
    uniformly in the interval [0,1]². They will be assigned the label 1, if they
    are withing the circle with radius "radius" and center point
    "(center_value, center_value, ..., center_value)".

    Args:
        N (int):                Number of Train/Test Data Points
        center_value (float):   Center Values of Circle
        radius (float):         Radius of Circle
    Returns:
        X_train (np.ndarray [Nx2]):     Training Samples
        Y_train (np.ndarray [Nx1]):     Training Labels
        X_test (np.ndarray [Nx2]):      Test Samples
        Y_test (np.ndarray [Nx1]):      Test Labels
    """

    # random generate the samples
    X = np.random.uniform(0, 1, (N*2, 2))
    # generate label why subtracting from the center and calculating norm
    Y = (np.linalg.norm(X-center_value, axis=1) <= radius).astype(int)

    return X[:N], Y[:N], X[N:], Y[N:]

def plot_data(X, Y):
    colors = {}
    plt.figure("Data Plot")
    plt.scatter(X[:,0], X[:,1], color=np.array(["red" if x == 0 else "blue" for x in Y]))
    plt.show()
