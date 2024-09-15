import sys
sys.path.append('../utils/')
sys.path.append('../paviaUTools/')
# sys.path.insert(1, '../utils')
# sys.path.insert(2, '../paviaUTools')

import matplotlib.pyplot as plt
from datasetLoader import datasetLoader
import os
import numpy as np
from whole_pipeline import whole_pipeline_all, whole_pipeline_all_euclidean, whole_pipeline_divided, whole_pipeline_divided_parallel, wasser_classify, wasser_hdd
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

is_normalize_each_band = True
method_label_patch='most_common'
M = 'euclidean'


dataset_name = 'Greding1'
factor = 9


WASSER_CALSSIFY = 1
WASSER_MHDD_HDD = 2
WASSER_MEUC_HDD = 3


def evaluate_wasser(c,k, X,y, factor, precomputed_distances,patched_labels, labels):
    torch.cuda.empty_cache()
    gc.collect()

    consts.CONST_C_PIXELS = c
    consts.CONST_K_PIXELS = k

    avg_acc_test = 0.0
    for i in range(reps):
        _,test_acc, _,_ =  wasser_hdd(X,y, factor, factor, is_normalize_each_band=is_normalize_each_band, method_label_patch=method_label_patch, random_seed=random_seeds[i], M=M, precomputed_pack=(precomputed_distances,patched_labels, labels))
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
        # Suggest a value for the hyperparameter within a given range
        c_hyperparameter = trial.suggest_float('c', self.min_c, self.max_c)
        k_hyperparameter = trial.suggest_int('k', self.min_k, self.max_k)

        print(f"working with k={k_hyperparameter} and c={c_hyperparameter}")
        
        tmp = torch.reshape(X, (X.shape[-1], -1)).float()
        distances_bands = torch.cdist(tmp, tmp)
        
        if is_normalize_each_band:
            X_tmp = HDD_HDE.normalize_each_band(X)
        else:
            X_tmp = X

        X_patches, _, _ = HDD_HDE.patch_data_class(X_tmp, factor, factor, y, method_label_patch)
        
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
        score = evaluate_wasser(c_hyperparameter, k_hyperparameter, self.X,self.y, self.factor, precomputed_distances,self.patched_labels, self.labels)
        
        return score


if __name__ == '__main__':
    print("RESULTS FOR WHDD!")
    
    
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
    
    print("shpae" , len(np.unique(y)))

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

