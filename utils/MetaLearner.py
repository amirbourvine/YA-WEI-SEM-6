from HDD_HDE import *
import torch


import itertools
def findMaxCombinations(tensor, evaluate, min_batch_size, max_batch_size, n):
    # Generate all possible not empty subsets
    all_subsets = []
    indices = range(tensor.shape[-1])
    for r in range(min_batch_size, max_batch_size + 1):
        subsets = torch.IntTensor(list(itertools.combinations(indices, r)))
        all_subsets.extend(subsets)
    
    # Evaluate each subset using the provided evaluate function
    evaluated_subsets = torch.FloatTensor([evaluate(torch.index_select(tensor, -1, subset)) for subset in all_subsets])

    # find the n largest evaluated subsets
    values, largest_subsets_indices = torch.topk(evaluated_subsets, n)

    return  values[:(largest_subsets_indices.shape[0])] ,[all_subsets[largest_subsets_indices[i].item()] for i in range(largest_subsets_indices.shape[0])]

class HDDOnBands:
    def run(tensor):
        tmp = torch.reshape(tensor, (tensor.shape[-1], -1)).float()

        distances = torch.cdist(tmp, tmp)

        return HDD_HDE.run_method(distances)
    
    def createUniformWeightedBatches(tensor):
        return torch.ones(tensor.shape[-1]), [[i] for i in range(tensor.shape[-1])]

    def createL1WeightedBatches(tensor):
        return torch.sum(HDDOnBands.run(tensor), axis=1), [[i] for i in range(tensor.shape[-1])]

    def createMinSimilarityBasedBatches(tensor, n):
        def evaluate(ten):
            return torch.norm(ten[:,:,0].float()-ten[:,:,1].float(), p=1)
        
        return findMaxCombinations(tensor, evaluate, 2, 2, n)
