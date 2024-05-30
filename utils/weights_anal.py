import matplotlib.pyplot as plt 
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def show_distances(distances):
    plt.imshow(distances)
    plt.colorbar()
    plt.show()

def show_weights(weights, is_normalized=True):
    x_data = np.arange(weights.shape[0])
    y_data = weights
    plt.plot(x_data, y_data)
    if is_normalized:
        plt.ylim((0,1))
    plt.show()

def find_levels(weights):
    levels = np.log(weights)
    return levels/(np.min(levels))



