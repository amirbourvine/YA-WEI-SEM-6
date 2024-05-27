from HDD_HDE import *
import torch

class HDDOnBands:
    def run(tensor):
        tmp = torch.reshape(tensor, (tensor.shape[-1], -1)).float()

        distances = torch.cdist(tmp, tmp)

        return HDD_HDE.run_method(distances)
    
    def createWeights_sumRows(tensor):
        return torch.sum(HDDOnBands.run(tensor), axis=1).cpu().numpy()
    
    def createWeights_Clusters(tensor):
        dist_mat = HDDOnBands.run(tensor)
        res = torch.cdist(dist_mat,dist_mat,p=1)
        