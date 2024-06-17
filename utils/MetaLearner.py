from HDD_HDE import *
import torch

from HDD_HDE import HDD_HDE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import random
from sklearn.cluster import AgglomerativeClustering
import numpy as np

def random_clusters(clusters_num, bands_num, random_seed=None):
    if random_seed is not None:
        random.seed(random_seed)
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

def size_weights(clusters):
    return torch.tensor([len(cluster) for cluster in clusters])

def sqrt_weights(clusters):
    return torch.tensor([np.sqrt(len(cluster)) for cluster in clusters])

def uniform_weights(clusters):
    return torch.ones(len(clusters))

def maxDiffWeights(tensor, clusters):
    weights = []
    for cluster in clusters:
        if len(cluster) == 1:
            weights.append(0)
        else:
            pairs = list(itertools.combinations(cluster, 2))
            pairs_diffs = [torch.sum(torch.abs(tensor[:, :, a] - tensor[:, :, b])) for (a,b) in pairs]
            weights.append(max(pairs_diffs))

    return torch.tensor([weights])

# norm = data_tmp / torch.max(data_tmp.norm(dim=1)[:, None],torch.tensor(eps,device=device))
# tmp[i,j,:,:] = 1 - torch.mm(norm, norm.transpose(0,1))
def cosine(tensor, eps):
    tensor[tensor<eps] = eps
    norm = tensor / tensor.norm(dim=1)[:, None]
    return 1 - torch.mm(norm, norm.transpose(0,1))

class HDDOnBands:
    def run(tensor, metric, factors_for_batch=None, eps=1e-8):
        if factors_for_batch is None:
            if metric!='euclidean' and metric!='cosine':
                print("ERROR- INVALID METRIC")
                return None
            tmp = torch.reshape(tensor, (tensor.shape[-1], -1)).float()

            if metric=='euclidean':
                distances = torch.cdist(tmp, tmp)
            elif metric=='cosine':
                distances = cosine(tmp, eps)
            
            return HDD_HDE.run_method(distances)
        else:
            rows_factor, cols_factor = factors_for_batch
            patched_data = HDD_HDE.patch_data_class(tensor, rows_factor, cols_factor, y=None, method_label_patch=None).float()
            # print(patched_data.shape) # torch.Size([88, 49, 7, 7, 103])
            tmp = torch.zeros(size=(patched_data.shape[0],patched_data.shape[1], patched_data.shape[-1], patched_data.shape[-1]))
            for i in range(tmp.shape[0]):
                for j in range(tmp.shape[1]):
                    data_tmp = patched_data[i,j,:,:, :]
                    data_tmp = torch.reshape(data_tmp, (data_tmp.shape[-1], -1))
                    if metric=='euclidean':
                        tmp[i,j,:,:] = torch.cdist(data_tmp, data_tmp)
                    elif metric=='cosine':
                        # tmp[i,j,:,:] = 1 - torch.nn.functional.cosine_similarity(data_tmp, data_tmp)
                        tmp[i,j,:,:] = cosine(data_tmp, eps)
            
            tmp = torch.reshape(tmp, shape=(tmp.shape[0]*tmp.shape[1], tmp.shape[2], tmp.shape[3]))
            distances = torch.linalg.norm(tmp, dim=(0))
            
            return distances
            

    
    #Partition component
    def createUniformWeightedBatches(tensor, random_seed=None, clusters_amount=None):
        if clusters_amount is None:
            return torch.ones(tensor.shape[-1]), [torch.tensor([i]) for i in range(tensor.shape[-1])]

        bands_num = tensor.shape[-1]
        clusters = random_clusters(clusters_amount, bands_num, random_seed=random_seed)
        weights = sqrt_weights(clusters)

        return weights, clusters
    
    def classicUnsurpervisedClustering(tensor, clusters_amount):
        distances = HDDOnBands.run(tensor, metric=METRIC_BANDS).cpu().numpy()
        clustering = AgglomerativeClustering(n_clusters = clusters_amount, metric="precomputed", linkage='average').fit(distances)
        clusters = [[] for i in range(clusters_amount)]
        for i, cluster in enumerate(clustering.labels_):
            clusters[cluster].append(i)
        weights = torch.tensor([len(cluster) for cluster in clusters])

        distances= distances

        aggClustering = AgglomerativeClustering(n_clusters=clusters_amount, metric="precomputed", linkage="average").fit(distances)
        clusters = [[] for i in range(clusters_amount)]
        for i, cluster in enumerate(aggClustering.labels_):
            clusters[cluster].append(i)
        clusters = [torch.tensor(cluster) for cluster in clusters]

        weights = sqrt_weights(clusters)
            
        return weights, clusters
    
    def regroupingUnsurpervisedClusters(tensor, clusters_amount):
        distances = HDDOnBands.run(tensor, metric= METRIC_BANDS)

        distances= distances.cpu().numpy()

        aggClustering = AgglomerativeClustering(n_clusters= clusters_amount, metric="precomputed", linkage="average").fit(distances)
        clustersBySimilarity = [[] for i in range(clusters_amount)]
        for i, cluster in enumerate(aggClustering.labels_):
            clustersBySimilarity[cluster].append(i)

        clusters = [[] for i in range(clusters_amount)]
        counter = 0
        for similarityCluster in clustersBySimilarity:
            for band in similarityCluster:
                clusters[counter % clusters_amount].append(band)
                counter = counter + 1

        clusters = [torch.tensor(cluster) for cluster in clusters]

        weights = sqrt_weights(clusters)
            
        return weights, clusters

    def createL1WeightedBatches(tensor, clusters_amount=None, normalize=True):
        res = normalize_weights(torch.sum(HDDOnBands.run(tensor, METRIC_BANDS), axis=1)).cpu().numpy()
        if not normalize:
            res = (torch.sum(HDDOnBands.run(tensor, ), axis=1)).cpu().numpy()
        return res, [torch.tensor([i]) for i in range(tensor.shape[-1])]

    #Evaluation Component
    def createMinSimilarityBasedBatches(tensor, n):
        def evaluate(ten):
            return torch.norm(ten[:,0].float()-ten[:,1].float(), p=1)
        
        return findMaxCombinations(HDDOnBands.run(tensor, METRIC_BANDS), evaluate, 2, 2, n)