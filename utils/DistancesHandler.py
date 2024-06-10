from consts import *
import numpy as np
import ot

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DistanceHandler:
    def __init__(self, method_type, distances_bands):
        self.method_type = method_type
        self.distances_bands = distances_bands
    

    def calc_distances(self, X_patches):
        if self.method_type==REGULAR_METHOD:
            X_patches_tmp = torch.reshape(X_patches, (-1, np.prod(X_patches.shape[2:])))
            del X_patches
            distances = torch.cdist(X_patches_tmp, X_patches_tmp)
            del X_patches_tmp

        elif self.method_type==MEAN_PATCH:
            X_patches_tmp = torch.mean(X_patches.float(), (2,3))

            del X_patches

            X_patches_tmp_tmp = (torch.max(X_patches_tmp, -1).indices)

            del X_patches_tmp

            X_patches = X_patches_tmp_tmp.reshape((X_patches_tmp_tmp.shape[0]*X_patches_tmp_tmp.shape[1],))

            del X_patches_tmp_tmp

            indices = torch.meshgrid(X_patches, X_patches)
            distances = self.distances_bands[indices]

            del indices
            del X_patches
        
        elif self.method_type==MEAN_DISTANCES:
            X_patches_tmp = (torch.max(X_patches, -1).indices).reshape((X_patches.shape[0]*X_patches.shape[1], X_patches.shape[2]*X_patches.shape[3]))

            del X_patches

            distances = torch.zeros((X_patches_tmp.shape[0],X_patches_tmp.shape[0]), device=device)

            for i in range(X_patches_tmp.shape[0]):
                for j in range(X_patches_tmp.shape[0]):
                    distances[i,j] = torch.mean(self.distances_bands[(X_patches_tmp[i,:], X_patches_tmp[j,:])])


            del X_patches_tmp

        elif self.method_type==WASSERSTEIN:
            X_patches_tmp = torch.reshape(X_patches, (-1, X_patches.shape[2], X_patches.shape[3], X_patches.shape[4]))
            del X_patches
            X_patches_vector = X_patches_tmp.cpu().numpy()
            del X_patches_tmp

            #Calculate the sum of each patch to generate its corresponding datapoint
            X_patches_vector = np.sum(X_patches_vector, axis=-2)
            X_patches_vector = np.sum(X_patches_vector, axis=-2)

            #Normalize the patches to disturbutions
            X_patches_vector = np.transpose(X_patches_vector)
            min_vals = X_patches_vector.min(axis=0, keepdims=True)
            max_vals = X_patches_vector.max(axis=0, keepdims=True)
            X_patches_vector = (X_patches_vector - min_vals) / (max_vals - min_vals)
            X_patches_vector = X_patches_vector / np.sum(X_patches_vector, axis=0)

            distances = np.zeros((X_patches_vector.shape[1],X_patches_vector.shape[1]))

            self.distances_bands = self.distances_bands.cpu().numpy()

            # Compute Wasserstein distance
            for i in range(X_patches_vector.shape[1]):
                print(i, " / ", X_patches_vector.shape[1])
                for j in range(X_patches_vector.shape[1]):
                    distances[i,j] = ot.emd2(X_patches_vector[:,i], X_patches_vector[:,j], self.distances_bands)

            distances = torch.tensor(distances, dtype=dist_dtype)
            distances = distances.to(device=device)

        return distances