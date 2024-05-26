from HDD_HDE import *
import torch

class HDDOnBands:
    def run(tensor):
        tmp = torch.reshape(tensor, (tensor.shape[-1], -1)).float()

        print("tmp.shape: ", tmp.shape)

        distances = torch.cdist(tmp, tmp)

        print("distances.shape: ", distances.shape)

        return HDD_HDE.run_method(distances)
    
    def createUniformWeightedBatches(tensor):
        return torch.ones(tensor.shape[-1]), [[i] for i in range(tensor.shape[-1])]

    def createL1WeightedBatches(tensor):
        return torch.sum(HDDOnBands.run(tensor), axis=1), [[i] for i in range(tensor.shape[-1])]
