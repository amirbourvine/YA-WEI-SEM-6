import consts
import numpy as np
import ot
import torch.multiprocessing as mp
import time

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cpu = torch.device("cpu")
class DistanceHandler:
    def __init__(self, method_type, distances_bands):
        self.method_type = method_type
        self.distances_bands = distances_bands
    
    def emd2_wrapper(vec1, vec2, distances_bands, i, j):
        return ot.emd2(vec1, vec2, distances_bands), i, j
    

    def calc_distances(self, X_patches):
        if self.method_type==consts.REGULAR_METHOD:
            X_patches_tmp = torch.reshape(X_patches, (-1, np.prod(X_patches.shape[2:]))).float()
            del X_patches

            if consts.METRIC_PIXELS=='euclidean':
                distances = torch.cdist(X_patches_tmp, X_patches_tmp)
            elif consts.METRIC_PIXELS=='cosine':
                norm = X_patches_tmp / X_patches_tmp.norm(dim=1)[:, None]
                distances = 1 - torch.mm(norm, norm.transpose(0,1))
            else:
                print("ERROR- INVALID METRIC")
                return None
            
            del X_patches_tmp

        elif self.method_type==consts.MEAN_PATCH:
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
        
        elif self.method_type==consts.MEAN_DISTANCES:
            X_patches_tmp = (torch.max(X_patches, -1).indices).reshape((X_patches.shape[0]*X_patches.shape[1], X_patches.shape[2]*X_patches.shape[3]))

            del X_patches

            distances = torch.zeros((X_patches_tmp.shape[0],X_patches_tmp.shape[0]), device=device)

            for i in range(X_patches_tmp.shape[0]):
                for j in range(X_patches_tmp.shape[0]):
                    distances[i,j] = torch.mean(self.distances_bands[(X_patches_tmp[i,:], X_patches_tmp[j,:])])


            del X_patches_tmp

        elif self.method_type==consts.WASSERSTEIN:
            X_patches_tmp = torch.reshape(X_patches, (-1, X_patches.shape[2], X_patches.shape[3], X_patches.shape[4]))
            del X_patches
            X_patches_vector = X_patches_tmp
            del X_patches_tmp

            #Calculate the sum of each patch to generate its corresponding datapoint
            X_patches_vector = torch.sum(X_patches_vector, dim=-2)
            X_patches_vector = torch.sum(X_patches_vector, dim=-2)

            #Normalize the patches to disturbutions
            X_patches_vector = torch.transpose(X_patches_vector,0,1)
            
            min_vals = X_patches_vector.min(dim=0, keepdims=True).values
            max_vals = X_patches_vector.max(dim=0, keepdims=True).values
            X_patches_vector = (X_patches_vector - min_vals) / (max_vals - min_vals)
            X_patches_vector = X_patches_vector / torch.sum(X_patches_vector, dim=0)

            distances = torch.zeros((X_patches_vector.shape[1],X_patches_vector.shape[1]), device=cpu)

            # # # parallel code section START
            print("start parallel code!", flush=True)
            
            # send all to cpu
            distances_bands = self.distances_bands.cpu().numpy().copy()
            X_patches_vector = X_patches_vector.cpu().numpy().copy()
            
            st = time.time()
            
            
            try:
                mp.set_start_method('spawn')
            except RuntimeError:
                print("MAY BE A PROBLEM: START_METHOD IS NOT SPAWN!")
            
            pool_size = consts.POOL_SIZE_CPU
            pool = mp.Pool(processes=pool_size)

            print(f"running with pool_size={pool_size}", flush=True)
            
            tup_list = []

            for i in range(X_patches_vector.shape[1]):
                for j in range(i+1, X_patches_vector.shape[1]):
                    tup = (X_patches_vector[:,i], X_patches_vector[:,j], distances_bands, i, j)
                    tup_list.append(tup)

            print(f"now starting calculate wasser", flush=True)
            
            for result in pool.starmap(DistanceHandler.emd2_wrapper, tup_list):
                res, i, j = result
                distances[i,j] = res
                distances[j, i] = res
                # print("done", flush=True)

            pool.close()  # no more tasks

            pool.join()  # wrap up current tasks

            print("PARALLEL WASSER TIME: ", time.time()-st, flush=True)
            
            
            distances.to(device)
            
            # parallel code section END

            # st = time.time()
            
            # for i in range(X_patches_vector.shape[1]):
            #     for j in range(i + 1, X_patches_vector.shape[1]):
            #         distances[i, j] = ot.emd2(X_patches_vector[:, i], X_patches_vector[:, j], distances_bands)
            #         distances[j, i] = distances[i, j] 

            # print("SERIAL WASSER TIME: ", time.time()-st)
            
            # IMPROVEMENT SECTION FOR EFFICIENCY **START**


            # distances_try = torch.zeros((X_patches_vector.shape[1],X_patches_vector.shape[1]), device=device)

            # st = time.time()


            # #put here the code for distances_try

            # print("IMPROVEMENT WASSER TIME: ", time.time()-st)

            
            # print(f"IS IMPROVEMENT GOOD? {torch.allclose(distances, distances_try)}")


            # IMPROVEMENT SECTION FOR EFFICIENCY **END**
        
        print("finished calculating distances", flush=True)
        return distances