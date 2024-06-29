"""
TO RUN: python3 wasser_classify.py > wasser_classify.txt

NOTES:
"""


import sys
import time
sys.path.append('../utils/')
sys.path.append('../paviaUTools/')
# sys.path.insert(1, '../utils')
# sys.path.insert(2, '../paviaUTools')

import matplotlib.pyplot as plt
from datasetLoader import datasetLoader
import os
import numpy as np
from whole_pipeline import whole_pipeline_all, whole_pipeline_divided, whole_pipeline_divided_parallel, wasser_classify
import torch
from plots import *
from weights_anal import *
from MetaLearner import HDDOnBands
from HDD_HDE import HDD_HDE
import DistancesHandler
import consts
import numpy as np
import pandas as pd

import gc

import warnings
warnings.filterwarnings("ignore")

torch.cuda.empty_cache()
gc.collect()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    parent_dir = os.path.join(os.getcwd(),"..")
    csv_path = os.path.join(parent_dir, 'datasets', 'paviaU.csv')
    gt_path = os.path.join(parent_dir, 'datasets', 'paviaU_gt.csv')
    dataset_name = 'paviaU'
    
    # csv_path = os.path.join(parent_dir, 'datasets', 'pavia.csv')
    # gt_path = os.path.join(parent_dir, 'datasets', 'pavia_gt.csv')
    # dataset_name = 'paviaCenter'
    
    # csv_path = os.path.join(parent_dir, 'datasets', 'KSC.csv')
    # gt_path = os.path.join(parent_dir, 'datasets', 'KSC_gt.csv')
    # dataset_name = 'KSC'
    

    dsl = datasetLoader(csv_path, gt_path)

    df = dsl.read_dataset(gt=False)
    X = np.array(df)
    X = X.reshape((610,340, 103))
    # X = X.reshape((1096, 715, 102))
    # X = X.reshape((512, 614, 176))

    df = dsl.read_dataset(gt=True)
    y = np.array(df)

    X = torch.from_numpy(X)
    y = torch.from_numpy(y)

    X = X.to(device)
    y = y.to(device)

    factor = 61
    is_normalize_each_band = True
    method_label_patch = 'most_common'

    distances_bands = HDDOnBands.run(X, metric=consts.METRIC_BANDS)
    distances_bands = distances_bands.to(device)

    if is_normalize_each_band:
        X_tmp = HDD_HDE.normalize_each_band(X)
    else:
        X_tmp = X
        
    X_patches, _, _= HDD_HDE.patch_data_class(X_tmp, factor, factor, y, method_label_patch)
    distance_handler = DistancesHandler.DistanceHandler(consts.WASSERSTEIN,distances_bands)
    
    num_reps = 1
    
    st = time.time()
    for i in range(num_reps):
        precomputed_distances = distance_handler.calc_distances(X_patches)
        
    print("AVG TIME: ", (time.time()-st)/num_reps)