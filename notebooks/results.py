import sys
sys.path.append('../utils/')
sys.path.append('../paviaUTools/')
# sys.path.insert(1, '../utils')
# sys.path.insert(2, '../paviaUTools')

import matplotlib.pyplot as plt
from datasetLoader import datasetLoader
import os
import numpy as np
from whole_pipeline import whole_pipeline_all, whole_pipeline_divided, whole_pipeline_divided_parallel
import torch
from plots import *
from weights_anal import *
from MetaLearner import HDDOnBands
from consts import *

import gc

if __name__ == '__main__':

    torch.cuda.empty_cache()
    gc.collect()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parent_dir = os.path.join(os.getcwd(),"..")
    csv_path = os.path.join(parent_dir, 'datasets', 'paviaU.csv')
    gt_path = os.path.join(parent_dir, 'datasets', 'paviaU_gt.csv')
    # csv_path = os.path.join(parent_dir, 'datasets', 'pavia.csv')
    # gt_path = os.path.join(parent_dir, 'datasets', 'pavia_gt.csv')

    dsl = datasetLoader(csv_path, gt_path)

    df = dsl.read_dataset(gt=False)
    X = np.array(df)
    X = X.reshape((610,340, 103))
    # X = X.reshape((1096, 715, 102))

    df = dsl.read_dataset(gt=True)
    y = np.array(df)

    X = torch.from_numpy(X)
    y = torch.from_numpy(y)

    X = X.to(device)
    y = y.to(device)

    reps = 10

    import random
    import numpy as np

    

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
    
    distances_bands = HDDOnBands.run(X)
    distances_bands = distances_bands.to(device)
    
    for method in [MEAN_PATCH, MEAN_DISTANCES]:
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        print("METHOD: ", method)

        for factor in[11,7,5,4]:
            avg_acc_train = 0.0
            avg_acc_test = 0.0
            for i in range(reps):
                train_acc,test_acc, test_preds,test_gt = whole_pipeline_all(X,y, factor, factor, is_normalize_each_band=False, method_label_patch='most_common', random_seed=random_seeds[i], method_type = method, distances_bands=distances_bands)
                avg_acc_train += train_acc/reps
                avg_acc_test += test_acc/reps

                print("iteration ", i, " DONE")

                torch.cuda.empty_cache()
                gc.collect()

            print("factor: ", factor)
            print("avg_acc_train: ", avg_acc_train)
            print("avg_acc_test: ", avg_acc_test)


    exit()

    #Partition componenets
    clusters_amounts = [5,25]

    #Random Partition component

    
    for clusters_amount in clusters_amounts:
        avg_acc_train = 0.0
        avg_acc_test = 0.0

        factor = 7
        if clusters_amount in [25]:
            factor = 11
        rows_factor = factor
        cols_factor = factor


        print("-------------------------")
        print("clusters amount: ", clusters_amount)
        print("-------------------------")

        for i in range(reps):
            weights, dist_batches = HDDOnBands.createUniformWeightedBatches(X, clusters_amount=clusters_amount, random_seed=random_seeds[i])
            train_acc,test_acc, test_preds,test_gt = whole_pipeline_divided_parallel(X,y, rows_factor, cols_factor, is_normalize_each_band=True, method_label_patch='most_common', weights=weights, distance_batches= dist_batches, random_seed=random_seeds[i])
            avg_acc_train += train_acc/reps
            avg_acc_test += test_acc/reps

            print("iteration ", i, " DONE")

            torch.cuda.empty_cache()
            gc.collect()

        print("Random Partition component")
        print("avg_acc_train: ", avg_acc_train)
        print("avg_acc_test: ", avg_acc_test)


        #Similarity based Partition component

        avg_acc_train = 0.0
        avg_acc_test = 0.0

        for i in range(reps):
            weights, dist_batches = HDDOnBands.classicUnsurpervisedClustering(X, clusters_amount=clusters_amount)
            train_acc,test_acc, test_preds,test_gt = whole_pipeline_divided_parallel(X,y, rows_factor, cols_factor, is_normalize_each_band=True, method_label_patch='most_common', weights=weights, distance_batches= dist_batches, random_seed=random_seeds[i])
            avg_acc_train += train_acc/reps
            avg_acc_test += test_acc/reps

            print("iteration ", i, " DONE")

            torch.cuda.empty_cache()
            gc.collect()

        print("Similarity based Partition component")
        print("avg_acc_train: ", avg_acc_train)
        print("avg_acc_test: ", avg_acc_test)


        # Regrouping Partition component

        avg_acc_train = 0.0
        avg_acc_test = 0.0

        for i in range(reps):
            weights, dist_batches = HDDOnBands.regroupingUnsurpervisedClusters(X, clusters_amount=clusters_amount)
            train_acc,test_acc, test_preds,test_gt = whole_pipeline_divided_parallel(X,y, rows_factor, cols_factor, is_normalize_each_band=True, method_label_patch='most_common', weights=weights, distance_batches= dist_batches, random_seed=random_seeds[i])
            avg_acc_train += train_acc/reps
            avg_acc_test += test_acc/reps

            print("iteration ", i, " DONE")

            torch.cuda.empty_cache()
            gc.collect()

        print("Regrouping Partition component")
        print("avg_acc_train: ", avg_acc_train)
        print("avg_acc_test: ", avg_acc_test)
