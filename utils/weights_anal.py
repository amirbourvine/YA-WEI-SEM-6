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


def show_structure(d_HDD):
  embedding = MDS(n_components=2, normalized_stress='auto', dissimilarity='precomputed')
  X_transformed = embedding.fit_transform(d_HDD)

  levels = np.zeros(X_transformed.shape[0])
  for i in range(levels.shape[0]):
    levels[i] = (i+1).bit_length()

  cmap = cm.Set1
  norm = colors.BoundaryNorm(np.arange(0.5, 6, 1), cmap.N)

  plt.scatter(X_transformed[:, 0], X_transformed[:, 1],c=levels, norm=norm, cmap = cmap, edgecolors='black',s=100)
  cbar = plt.colorbar(ticks=np.linspace(1, 5, 5))
  cbar.set_label('level')
  plt.show()



