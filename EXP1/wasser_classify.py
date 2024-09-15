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

reps = 100
is_normalize_each_band = True
method_label_patch='most_common'

dataset_name = 'Greding1'
factor = 9
M = 'euclidean'

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
    if dataset_name=='Botswana':
        csv_path = os.path.join(parent_dir, 'datasets', 'Botswana.csv')
        gt_path = os.path.join(parent_dir, 'datasets', 'Botswana_gt.csv')
        new_shape = (1476, 256, 145)
    if dataset_name=='Greding1':
        csv_path = os.path.join(parent_dir, 'datasets/new_dataset', 'Greding_Village1_refl.csv')
        gt_path = os.path.join(parent_dir, 'datasets/new_dataset', 'Greding_Village1_refl_gt.csv')
        new_shape = (670, 606, 126)
    
        
    dsl = datasetLoader(csv_path, gt_path)

    df = dsl.read_dataset(gt=False)
    X = np.array(df)
    X = X.reshape(new_shape)
    
    if dataset_name=='Greding1':
        X = X[:, :-1, :]
        new_shape = (670, 605, 126)

    df = dsl.read_dataset(gt=True)
    y = np.array(df)

    X = torch.from_numpy(X)
    y = torch.from_numpy(y)

    X = X.to(device)
    y = y.to(device)

    random_seeds = [-923723872, 883017324, 531811554, 2047094521, 1767143556, 112000582, -1699501351, -2096286485, -1079138285,-424805109,
    -288375640,    23449114,  1300214163, -1806192332,   417479967,  1846505308,
  1208973297,   735097938,  1938870395,  1454714989,  1194339285,   369373451,
  2058090566,   937472615,  -590811606,   512126150,  -985111295,   538622725,
   400515189,   661801117, -1024452399,  -968082218,  -91103061,  -771222861,
 -1925172284, -1742055560,  1405414557,  1286983770,  1343625340,   865065231,
   853534880,  -879759532,   995293658,  1325554155, -2096944877,  2032312566,
  1607190003,  -908097366, -1675790208,   496923971, -1185776576,  1027683730,
  1512566547,  -642847140,  1177796624,   345850328,   879159630,  2096252488,
   974939630, -1291733131, -2024129948, -1422386738, 1134111330,  1626041002,
  1838397252, -1027363346,   326183661,  1593593847,  -784898930, -1949679173,
  1989361884,   327042844, -2041465320,   444751788,  1860415771,  2008447136,
  -608904847,  -894178555,   -20301812,   631326188,  -120259494, -1520800888,
  -420876687, -1516514246,   886966222,  1660631783,  1346817148, -1072430167,
  1453994541,  1095205151,    92409430,  -298728788, -1271582853,  1536200765,
  2037508827,   829688693, -1892694653,  -709346213,  -485469174,   645700526]


    # distances_bands_hdd = HDDOnBands.run(X, consts.METRIC_BANDS, None)
    # distances_bands_hdd = distances_bands_hdd.to(device)

    tmp = torch.reshape(X, (X.shape[-1], -1)).float()
    distances_bands_euc = torch.cdist(tmp, tmp)
    distances_bands_euc = distances_bands_euc.to(device)
    
    print(f"working with factor={factor} on device={device}", flush=True)
    print(f"*******************RESULTS OF M={M}************************", flush=True)
    distances_bands = distances_bands_euc

    poss_file_name = f"wassers/wasser_{M}_{factor}_{dataset_name}"
    
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

    if not os.path.isfile(poss_file_name):
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
