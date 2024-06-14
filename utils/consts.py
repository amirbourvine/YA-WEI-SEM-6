import torch

CONST_K = 19
CONST_C = 5
TOL = 1e-6
ALPHA = 0.5
N_NEIGHBORS = 1
TEST_SIZE = 0.2
APPLY_2_NORM =True
dist_dtype = torch.float32 #for resolution adjustments
POOL_SIZE_WASSERSTEIN = 30
POOL_SIZE_HDD = 3

METRIC_BANDS = 'cosine'
METRIC_PIXELS = 'euclidean'

REGULAR_METHOD = 0
MEAN_PATCH = 1
MEAN_DISTANCES = 2
WASSERSTEIN = 3
