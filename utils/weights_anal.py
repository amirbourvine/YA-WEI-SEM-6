import matplotlib.pyplot as plt 
import numpy as np
from sklearn.manifold import MDS
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from matplotlib import colors


def show_distances(distances):
    plt.imshow(distances.cpu().numpy())
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

def show_structure(distances):
    embedding = MDS(n_components=2, normalized_stress='auto', dissimilarity='precomputed')
    X_transformed = embedding.fit_transform(distances.cpu().numpy())

    levels = np.zeros(X_transformed.shape[0])
    max_level = 0
    for i in range(levels.shape[0]):
        levels[i] = (i+1).bit_length()
        max_level = max(levels[i],max_level)

    max_level = int(max_level)
    cmap = cm.Set1
    norm = colors.BoundaryNorm(np.arange(0.5, max_level+1, 1), cmap.N)

    plt.scatter(X_transformed[:, 0], X_transformed[:, 1],c=levels, norm=norm, cmap = cmap, edgecolors='black',s=100)
    cbar = plt.colorbar(ticks=np.linspace(1, max_level, max_level))
    cbar.set_label('level')
    plt.show()



