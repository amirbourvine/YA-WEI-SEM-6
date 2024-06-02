from HDD_HDE import *
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import random
def random_clusters(clusters_num, bands_num):
    buffers = sorted(random.sample(range(1, bands_num), clusters_num - 1))
    list = []
    for i in range(len(buffers) - 1):
        list.append(torch.tensor(np.arange(buffers[i], buffers[i + 1])))
    return  [torch.tensor(np.arange(0, buffers[0]))] + list + [torch.tensor(np.arange(buffers[-1], bands_num))]

import itertools
def findMaxCombinations(tensor, evaluate, min_batch_size, max_batch_size, n):
    # Generate all possible not empty subsets
    all_subsets = []
    indices = range(tensor.shape[-1])
    for r in range(min_batch_size, max_batch_size + 1):
        subsets = torch.IntTensor(list(itertools.combinations(indices, r))).to(device)
        all_subsets.extend(subsets)
    
    # Evaluate each subset using the provided evaluate function
    evaluated_subsets = torch.FloatTensor([evaluate(torch.index_select(tensor, -1, subset)) for subset in all_subsets])

    # find the n largest evaluated subsets
    values, largest_subsets_indices = torch.topk(evaluated_subsets, n)

    return  normalize_weights(values[:(largest_subsets_indices.shape[0])]).cpu().numpy() ,[all_subsets[largest_subsets_indices[i].item()] for i in range(largest_subsets_indices.shape[0])]

def normalize_weights(weights):
    return torch.nn.functional.normalize(weights, p=1.0, dim = 0)


class HDDOnBands:
    def run(tensor):
        tmp = torch.reshape(tensor, (tensor.shape[-1], -1)).float()

        distances = torch.cdist(tmp, tmp)

        return HDD_HDE.run_method(distances)
    
    def createUniformWeightedBatches(tensor, clusters_amount=None):
        if clusters_amount is None:
            return torch.ones(tensor.shape[-1]), [torch.tensor([i]) for i in range(tensor.shape[-1])]

        bands_num = tensor.shape[-1]
        clusters = random_clusters(clusters_amount, bands_num)
        weights = torch.tensor([len(cluster) for cluster in clusters])

        return weights, clusters

    def createL1WeightedBatches(tensor, clusters_amount=None, normalize=True):
        res = normalize_weights(torch.sum(HDDOnBands.run(tensor), axis=1)).cpu().numpy()
        if not normalize:
            res = (torch.sum(HDDOnBands.run(tensor), axis=1)).cpu().numpy()
        return res, [torch.tensor([i]) for i in range(tensor.shape[-1])]

    def createMinSimilarityBasedBatches(tensor, n):
        def evaluate(ten):
            return torch.norm(ten[:,0].float()-ten[:,1].float(), p=1)
        
        return findMaxCombinations(HDDOnBands.run(tensor), evaluate, 2, 2, n)