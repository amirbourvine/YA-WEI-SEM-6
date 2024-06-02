import matplotlib.pyplot as plt 
import numpy as np
from sklearn.manifold import MDS
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from matplotlib import colors
import networkx as nx
import pydot
from networkx.drawing.nx_pydot import graphviz_layout

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

def plot_tree(distances):
    G = nx.from_numpy_array(distances) 
    T = nx.minimum_spanning_tree(G)

    # Visualize the graph and the minimum spanning tree
    am = nx.adjacency_matrix(T)

    find_levels(am)

    plt.spy(am, precision=0.1, markersize=5)
    plt.show()

    plt.figure(figsize=(100,100))
    pos = graphviz_layout(T, prog="twopi")
    nx.draw(T, pos, node_size=6000)
    plt.show()


    # pos = nx.spiral_layout(T,scale=2)
    # nx.draw(T,pos)

def find_levels(am):
    am = am.toarray()
    am = (am!=0)
    num_nei = np.sum(am,axis=1)
    front_indices = np.where(num_nei==1).tolist()
    closed = []

    levels_list_old = np.ones(num_nei.shape[0])
    levels_list_new = np.zeros(num_nei.shape[0])
    
    while(np.all((levels_list_new-levels_list_old)==0)):
        levels_list_old = levels_list_new.copy()

        front_indices_tmp = front_indices.copy()
        for i in front_indices:

            nei = np.where((am[i,:])==1).tolist()
            for j in nei:
                
                if (j not in closed) and (levels_list_old[i]+1>levels_list_old[j]):
                    levels_list_new[j] = levels_list_old[i]+1
                    if j not in front_indices:
                        front_indices_tmp.append(j)


            front_indices_tmp.remove(i)
            closed.append(i)
        
        front_indices = front_indices_tmp
                

        








