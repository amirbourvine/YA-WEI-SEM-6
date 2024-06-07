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

import gc
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

factor = 7
rows_factor = factor
cols_factor = factor

X = torch.from_numpy(X)
y = torch.from_numpy(y)

X = X.to(device)
y = y.to(device)

reps = 10

import random
import numpy as np

# 32-bit integer
int32_max = np.iinfo(np.int32).max
int32_min = np.iinfo(np.int32).min

random_seeds = [random.randint(int32_min, int32_max) for _ in range (reps)]

#Partition componenets
# cluster_amounts = [12, 16, 25, 30]
clusters_amounts = [2, 2]

import gc
from MetaLearner import HDDOnBands

#Random Partition component

avg_acc_train = 0.0
avg_acc_test = 0.0

for clusters_amount in clusters_amounts:
    print("-------------------------")
    print("clusters amount: ", clusters_amount)
    print("-------------------------")


    print("********************Random Partition component********************")
    for i in range(reps):
        weights, dist_batches = HDDOnBands.createUniformWeightedBatches(X, clusters_amount=clusters_amount, random_seed=random_seeds[i])
        train_acc,test_acc, test_preds,test_gt = whole_pipeline_divided_parallel(X,y, rows_factor, cols_factor, is_normalize_each_band=True, method_label_patch='most_common', weights=weights, distance_batches= dist_batches, random_seed=random_seeds[i])
        avg_acc_train += train_acc/reps
        avg_acc_test += test_acc/reps

        print("iteration ", i, " stats: ")
        print("train_acc: ", train_acc)
        print("test_acc: ", test_acc)

    print("avg_acc_train: ", avg_acc_train)
    print("avg_acc_test: ", avg_acc_test)


    #Similarity based Partition component

    avg_acc_train = 0.0
    avg_acc_test = 0.0
    print("********************Similarity based Partition component********************")
    for i in range(reps):
        weights, dist_batches = HDDOnBands.classicUnsurpervisedClustering(X, clusters_amount=clusters_amount)
        train_acc,test_acc, test_preds,test_gt = whole_pipeline_divided_parallel(X,y, rows_factor, cols_factor, is_normalize_each_band=True, method_label_patch='most_common', weights=weights, distance_batches= dist_batches, random_seed=random_seeds[i])
        avg_acc_train += train_acc/reps
        avg_acc_test += test_acc/reps

        print("iteration ", i, " stats: ")
        print("train_acc: ", train_acc)
        print("test_acc: ", test_acc)

    print("avg_acc_train: ", avg_acc_train)
    print("avg_acc_test: ", avg_acc_test)


    #Regrouping Partition component

    avg_acc_train = 0.0
    avg_acc_test = 0.0

    print("********************Regrouping Partition component********************")

    for i in range(reps):
        weights, dist_batches = HDDOnBands.regroupingUnsurpervisedClusters(X, clusters_amount=clusters_amount)
        train_acc,test_acc, test_preds,test_gt = whole_pipeline_divided_parallel(X,y, rows_factor, cols_factor, is_normalize_each_band=True, method_label_patch='most_common', weights=weights, distance_batches= dist_batches, random_seed=random_seeds[i])
        avg_acc_train += train_acc/reps
        avg_acc_test += test_acc/reps

        print("iteration ", i, " stats: ")
        print("train_acc: ", train_acc)
        print("test_acc: ", test_acc)

    print("avg_acc_train: ", avg_acc_train)
    print("avg_acc_test: ", avg_acc_test)
