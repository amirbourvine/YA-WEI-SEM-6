from consts import *
import numpy as np

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
        
        return distances