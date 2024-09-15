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

is_normalize_each_band = True
method_label_patch='most_common'

factor = 9

def evaluate(c,k, X,y, factor):
    torch.cuda.empty_cache()
    gc.collect()

    consts.CONST_C_PIXELS = c
    consts.CONST_K_PIXELS = k

    avg_acc_test = 0.0
    for i in range(reps):
        _,test_acc, _,_ = whole_pipeline_all(X,y, factor, factor, is_normalize_each_band=is_normalize_each_band, method_label_patch=method_label_patch, random_seed=random_seeds[i])
        
        avg_acc_test += test_acc/reps

    score = avg_acc_test
    return score


class Objective:
    def __init__(self, min_c, max_c, min_k, max_k, X,y, factor,patched_labels, labels, X_patches):
        # Hold this implementation specific arguments as the fields of the class.
        self.min_c = min_c
        self.max_c = max_c
        self.min_k = min_k
        self.max_k = max_k
        self.X = X
        self.y = y
        self.factor = factor
        self.patched_labels = patched_labels
        self.labels = labels
        self.X_patches = X_patches
        
    def __call__(self, trial):
        dataset_name = 'paviaU'
        
        # Suggest a value for the hyperparameter within a given range
        c_hyperparameter = trial.suggest_float('c', self.min_c, self.max_c)
        k_hyperparameter = trial.suggest_int('k', self.min_k, self.max_k)

        print(f"working with k={k_hyperparameter} and c={c_hyperparameter}")
        
        if is_normalize_each_band:
            X_tmp = HDD_HDE.normalize_each_band(X)
        else:
            X_tmp = X

        X_patches, _, _ = HDD_HDE.patch_data_class(X_tmp, factor, factor, y, method_label_patch)

        distance_handler = DistancesHandler.DistanceHandler(consts.REGULAR_METHOD)
        precomputed_distances = distance_handler.calc_distances(X_patches)

        precomputed_distances = precomputed_distances.to(device)
        score = evaluate(c_hyperparameter, k_hyperparameter, self.X,self.y, self.factor)
        
        return score


if __name__ == '__main__':
    parent_dir = os.path.join(os.getcwd(),"..")
    
    # csv_path = os.path.join(parent_dir, 'datasets', 'paviaU.csv')
    # gt_path = os.path.join(parent_dir, 'datasets', 'paviaU_gt.csv')
    # dataset_name = 'paviaU'
    
    # csv_path = os.path.join(parent_dir, 'datasets', 'pavia.csv')
    # gt_path = os.path.join(parent_dir, 'datasets', 'pavia_gt.csv')
    # dataset_name = 'paviaCenter'
    
    csv_path = os.path.join(parent_dir, 'datasets', 'KSC.csv')
    gt_path = os.path.join(parent_dir, 'datasets', 'KSC_gt.csv')
    dataset_name = 'KSC'
    
    dsl = datasetLoader(csv_path, gt_path)

    df = dsl.read_dataset(gt=False)
    X = np.array(df)
    # X = X.reshape((610,340, 103))
    # X = X.reshape((1096, 715, 102))
    X = X.reshape((512, 614, 176))

    df = dsl.read_dataset(gt=True)
    y = np.array(df)

    X = torch.from_numpy(X)
    y = torch.from_numpy(y)

    X = X.to(device)
    y = y.to(device)
        
    if is_normalize_each_band:
        X_tmp = HDD_HDE.normalize_each_band(X)
    else:
        X_tmp = X


    X_patches, patched_labels, labels= HDD_HDE.patch_data_class(X_tmp, factor, factor, y, method_label_patch)
    
    
    # Create a study object and specify the direction of optimization
    study = optuna.create_study(direction='maximize')

    min_c = 0.1
    max_c = 10.0
    min_k = 4
    max_k = 16

    # Start the optimization
    study.optimize(Objective(min_c, max_c, min_k, max_k, X,y, factor,patched_labels, labels, X_patches), n_trials=100)

    # Print the best hyperparameter and its corresponding score
    print("Best c: ", study.best_params['c'])
    print("Best k: ", study.best_params['k'])
    print("Best Score: ", study.best_value)

