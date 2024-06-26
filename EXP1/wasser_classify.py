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
    # csv_path = os.path.join(parent_dir, 'datasets', 'paviaU.csv')
    # gt_path = os.path.join(parent_dir, 'datasets', 'paviaU_gt.csv')
    # csv_path = os.path.join(parent_dir, 'datasets', 'pavia.csv')
    # gt_path = os.path.join(parent_dir, 'datasets', 'pavia_gt.csv')
    csv_path = os.path.join(parent_dir, 'datasets', 'KSC.csv')
    gt_path = os.path.join(parent_dir, 'datasets', 'KSC_gt.csv')

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

    for M in ['hdd', 'euclidean']:
        print(f"*******************RESULTS OF M={M}************************")
        if M=='hdd':
            distances_bands = distances_bands_hdd
        elif M=='euclidean':
            distances_bands = distances_bands_euc

        for factor in [11,9,7,5]:
            
            if is_normalize_each_band:
                X_tmp = HDD_HDE.normalize_each_band(X)
            else:
                X_tmp = X
            X_patches, _, _= HDD_HDE.patch_data_class(X_tmp, factor, factor, y, method_label_patch)
            distance_handler = DistancesHandler.DistanceHandler(consts.WASSERSTEIN,distances_bands)
            precomputed_distances = distance_handler.calc_distances(X_patches)

            if save_to_csv:
                df = pd.DataFrame(precomputed_distances.cpu().numpy())
                df.to_csv(f"wasser_{M}_{factor}_{csv_path}",index=False)

            avg_acc_train = 0.0
            avg_acc_test = 0.0
            for i in range(reps):
                train_acc,test_acc, test_preds,test_gt = wasser_classify(X,y, factor, factor, is_normalize_each_band=is_normalize_each_band, method_label_patch=method_label_patch, random_seed=random_seeds[i], M=M, precomputed_distances=precomputed_distances)
                avg_acc_train += train_acc/reps
                avg_acc_test += test_acc/reps

                print("iteration ", i, " DONE")

                torch.cuda.empty_cache()
                gc.collect()

            print("factor: ", factor)
            print("avg_acc_train: ", avg_acc_train)
            print("avg_acc_test: ", avg_acc_test)
