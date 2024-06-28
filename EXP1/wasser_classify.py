"""
TO RUN: python3 wasser_classify.py > wasser_classify.txt

NOTES:
"""


import sys
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

    reps = 10

    random_seeds = [-923723872,
    883017324,
    531811554,
    2047094521,
    1767143556,
    112000582,
    -1699501351,
    -2096286485,
    -1079138285,
    -424805109]

    distances_bands_hdd = HDDOnBands.run(X, consts.METRIC_BANDS, None)
    distances_bands_hdd = distances_bands_hdd.to(device)

    tmp = torch.reshape(X, (X.shape[-1], -1)).float()
    distances_bands_euc = torch.cdist(tmp, tmp)
    distances_bands_euc = distances_bands_euc.to(device)

    is_normalize_each_band = True
    method_label_patch='most_common'
    save_to_csv = True

    task_id = int(sys.argv[1])
    ind_factor = task_id % 4
    ind_M = 0 if task_id<4 else 1
    factors = [11,9,7,5]
    Ms = ['hdd', 'euclidean']
    M = Ms[ind_M]
    factor = factors[ind_factor]
    
    print(f"worker {task_id} is working with factor={factor} on device={device}", flush=True)
    print(f"*******************RESULTS OF M={M}************************", flush=True)
    if M=='hdd':
        distances_bands = distances_bands_hdd
    elif M=='euclidean':
        distances_bands = distances_bands_euc

    poss_file_name = f"wasser_{M}_{factor}_{dataset_name}"
    
    if is_normalize_each_band:
        X_tmp = HDD_HDE.normalize_each_band(X)
    else:
        X_tmp = X

    X_patches, patched_labels, labels= HDD_HDE.patch_data_class(X_tmp, factor, factor, y, method_label_patch)
        
    if os.path.isfile(poss_file_name):
        print("USING SAVED PRECOMPUTED DISTANCES!", flush=True)
        df = pd.read_csv(poss_file_name)
        precomputed_distances = torch.Tensor(df.to_numpy())
          
    else:
        distance_handler = DistancesHandler.DistanceHandler(consts.WASSERSTEIN,distances_bands)
        precomputed_distances = distance_handler.calc_distances(X_patches)

    if save_to_csv and not os.path.isfile(poss_file_name):
        df = pd.DataFrame(precomputed_distances.cpu().numpy())
        df.to_csv(poss_file_name,index=False)

    avg_acc_train = 0.0
    avg_acc_test = 0.0
    for i in range(reps):
        train_acc,test_acc, test_preds,test_gt = wasser_classify(X,y, factor, factor, is_normalize_each_band=is_normalize_each_band, method_label_patch=method_label_patch, random_seed=random_seeds[i], M=M, precomputed_pack=(precomputed_distances,patched_labels, labels))
        avg_acc_train += train_acc/reps
        avg_acc_test += test_acc/reps

        print("iteration ", i, " DONE", flush=True)

        torch.cuda.empty_cache()
        gc.collect()

    print("factor: ", factor, flush=True)
    print("avg_acc_train: ", avg_acc_train, flush=True)
    print("avg_acc_test: ", avg_acc_test, flush=True)
