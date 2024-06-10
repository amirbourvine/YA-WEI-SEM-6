import torch

CONST_K = 19
CONST_C = 5
TOL = 1e-6
ALPHA = 0.5
N_NEIGHBORS = 1
TEST_SIZE = 0.2
APPLY_2_NORM =True
dist_dtype = torch.float32 #for resolution adjustments
POOL_SIZE = 3
METRIC = 'euclidean'

REGULAR_METHOD = 0
MEAN_PATCH = 1
MEAN_DISTANCES = 2
WASSERSTEIN = 3
