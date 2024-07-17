import sys
sys.path.append('../utils/')
sys.path.append('../paviaUTools/')
# sys.path.insert(1, '../utils')
# sys.path.insert(2, '../paviaUTools')

import matplotlib.pyplot as plt
from datasetLoader import datasetLoader
import os
import numpy as np
from whole_pipeline import whole_pipeline_all, whole_pipeline_divided, whole_pipeline_divided_parallel, wasser_classify, wasser_hdd
import torch
from plots import *
from weights_anal import *
from MetaLearner import HDDOnBands
from HDD_HDE import HDD_HDE
import DistancesHandler
import consts
import numpy as np
import pandas as pd
import optuna
import gc

import warnings
warnings.filterwarnings("ignore")

torch.cuda.empty_cache()
gc.collect()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

is_normalize_each_band = True
method_label_patch='most_common'
M = 'euclidean'

reps = 2
dataset_name = 'paviaU'
factor = 11


if __name__ == '__main__':
    
    parent_dir = os.path.join(os.getcwd(),"..")
    
    if dataset_name=='paviaU':
        csv_path = os.path.join(parent_dir, 'datasets', 'paviaU.csv')
        gt_path = os.path.join(parent_dir, 'datasets', 'paviaU_gt.csv')
        new_shape = (610,340, 103)
    if dataset_name=='pavia':
        csv_path = os.path.join(parent_dir, 'datasets', 'pavia.csv')
        gt_path = os.path.join(parent_dir, 'datasets', 'pavia_gt.csv')
        new_shape = (1096, 715, 102)
    if dataset_name=='KSC':
        csv_path = os.path.join(parent_dir, 'datasets', 'KSC.csv')
        gt_path = os.path.join(parent_dir, 'datasets', 'KSC_gt.csv')
        new_shape = (512, 614, 176)
        
    dsl = datasetLoader(csv_path, gt_path)

    df = dsl.read_dataset(gt=False)
    X = np.array(df)
    X = X.reshape(new_shape)

    df = dsl.read_dataset(gt=True)
    y = np.array(df)

    X = torch.from_numpy(X)
    y = torch.from_numpy(y)

    X = X.to(device)
    y = y.to(device)
    
    print(f"worker is working with factor={factor} on device={device} and validating *PIXELS* HYPERPARAMS ON *{dataset_name}*", flush=True)
    
    if is_normalize_each_band:
        X_tmp = HDD_HDE.normalize_each_band(X)
    else:
        X_tmp = X


    X_patches, patched_labels, labels= HDD_HDE.patch_data_class(X_tmp, factor, factor, y, method_label_patch)
    
    if M=='euclidean':
        tmp = torch.reshape(X, (X.shape[-1], -1)).float()
        distances_bands = torch.cdist(tmp, tmp)
    else:
        print("ERROR- only euclidean between bands supported!")
        exit(1)
    
    poss_file_name = f"wassers/wasser_{M}_{factor}_{dataset_name}"
    
    if os.path.isfile(poss_file_name):
        print("USING SAVED PRECOMPUTED DISTANCES!", flush=True)
        df = pd.read_csv(poss_file_name)
        precomputed_distances = torch.Tensor(df.to_numpy())
    else:
        distance_handler = DistancesHandler.DistanceHandler(consts.WASSERSTEIN,distances_bands)
        precomputed_distances = distance_handler.calc_distances(X_patches)
        df = pd.DataFrame(precomputed_distances.cpu().numpy())
        df.to_csv(poss_file_name,index=False)
    
    precomputed_distances = precomputed_distances.to(device)
    

    avg_acc_test = 0.0
    avg_acc_train = 0.0
    for i in range(reps):
        torch.cuda.empty_cache()
        gc.collect()
        train_acc,test_acc, _,_ = wasser_hdd(X,y, factor, factor, is_normalize_each_band=is_normalize_each_band, method_label_patch=method_label_patch, random_seed=random_seeds[i], M=M, precomputed_pack=(precomputed_distances,patched_labels, labels))
        avg_acc_test += test_acc/reps
        avg_acc_train += train_acc/reps


    print("FINAL RESULTS:")
    print("avg train_acc=", train_acc)
    print("avg test_acc=", test_acc)
