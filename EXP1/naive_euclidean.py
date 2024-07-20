
import sys
sys.path.append('../utils/')
sys.path.append('../paviaUTools/')
# sys.path.insert(1, '../utils')
# sys.path.insert(2, '../paviaUTools')

import matplotlib.pyplot as plt
from datasetLoader import datasetLoader
import os
import numpy as np
from whole_pipeline import whole_pipeline_all, whole_pipeline_divided, whole_pipeline_divided_parallel, wasser_classify, whole_pipeline_all_euclidean
import torch
from plots import *
from weights_anal import *
from MetaLearner import HDDOnBands
from HDD_HDE import HDD_HDE
import DistancesHandler
import consts
import numpy as np

import gc

import warnings
warnings.filterwarnings("ignore")

torch.cuda.empty_cache()
gc.collect()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

reps = 10
is_normalize_each_band = True
method_label_patch='most_common'


factor = 9
dataset_name = 'pavia'


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
     
    avg_acc_train = 0.0
    avg_acc_test = 0.0
    for i in range(reps):
        train_acc,test_acc, test_preds,test_gt = whole_pipeline_all_euclidean(X,y, factor, factor, is_normalize_each_band=is_normalize_each_band, method_label_patch=method_label_patch, random_seed=random_seeds[i])
        avg_acc_train += train_acc/reps
        avg_acc_test += test_acc/reps

        print("iteration ", i, " DONE")

        torch.cuda.empty_cache()
        gc.collect()

    print("factor: ", factor)
    print("avg_acc_train: ", avg_acc_train)
    print("avg_acc_test: ", avg_acc_test)
