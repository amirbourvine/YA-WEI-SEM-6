import numpy as np
import torch
from consts import CONST_K,ALPHA,TOL,CONST_C, N_NEIGHBORS, POOL_SIZE
import time
from HDD_HDE import *
from PaviaClassifier import *
import torch.multiprocessing as mp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cpu = torch.device("cpu")

def whole_pipeline_all(X,y, rows_factor, cols_factor, is_normalize_each_band=True, method_label_patch='center'):
        my_HDD_HDE = HDD_HDE(X,y, rows_factor, cols_factor, is_normalize_each_band, method_label_patch)

        print("XXXXXXX IN METHOD XXXXXXXXX")
        st = time.time()

        d_HDD, labels_padded, num_patches_in_row,y_patches = my_HDD_HDE.calc_hdd()

        print("WHOLE METHOD TIME: ", time.time()-st, flush=True)
        st = time.time()

        print("XXXXXXX IN CLASSIFICATION XXXXXXXXX")

        y_patches = y_patches.int()
        
        if torch.cuda.is_available():
            d_HDD = d_HDD.cpu()
            y_patches = y_patches.cpu()
            labels_padded = labels_padded.cpu()

        clf = PaviaClassifier(d_HDD.numpy(), y_patches.numpy(), N_NEIGHBORS, labels_padded.numpy(), rows_factor, cols_factor, num_patches_in_row, is_divided=False)

        clf.classify()

        print("WHOLE CLASSIFICATION TIME: ", time.time()-st)





def whole_pipeline_divided(X,y, rows_factor, cols_factor, is_normalize_each_band=True, method_label_patch='center', weights = None):
    st = time.time()
    
    num_patches = int(np.ceil(X.shape[0]/rows_factor)*np.ceil(X.shape[1]/cols_factor))

    distance_mat_arr = torch.zeros((X.shape[-1],num_patches,num_patches), device=device)
    for i in range(X.shape[-1]):
        X_curr = torch.reshape(X[:,:,i], (X.shape[0],X.shape[1],1))
        my_HDD_HDE = HDD_HDE(X_curr,y, rows_factor, cols_factor, is_normalize_each_band, method_label_patch)
        d_HDD, labels_padded, num_patches_in_row,y_patches = my_HDD_HDE.calc_hdd()
        distance_mat_arr[i,:,:] = d_HDD

        if i!=X.shape[-1]-1:
            del X_curr
            del d_HDD
            del labels_padded
            del y_patches
        
        print(f"DONE ITER. #{i+1} of {X.shape[-1]}")


    print("TOTAL TIME FOR METHOD: ", time.time()-st)

    y_patches = y_patches.int()

    if torch.cuda.is_available():
        distance_mat_arr = distance_mat_arr.cpu()
        y_patches = y_patches.cpu()
        labels_padded = labels_padded.cpu()

    clf = PaviaClassifier(distance_mat_arr.numpy(), y_patches.numpy(), N_NEIGHBORS, labels_padded.numpy(), rows_factor, cols_factor, num_patches_in_row, is_divided=True, weights=weights)

    clf.classify()



def whole_pipeline_divided_parallel(X,y, rows_factor, cols_factor, is_normalize_each_band=True, method_label_patch='center', weights = None):
    st = time.time()
    
    mp.set_start_method('spawn')
    pool_size =  POOL_SIZE if torch.cuda.is_available() else mp.cpu_count() * 2
    pool = mp.Pool(processes=pool_size)


    num_patches = int(np.ceil(X.shape[0]/rows_factor)*np.ceil(X.shape[1]/cols_factor))
    distance_mat_arr = torch.zeros((X.shape[-1],num_patches,num_patches), device=device)

    tup_list = []
    for i in range(X.shape[-1]):
        X_curr = X[:,:,i].reshape((X.shape[0],X.shape[1],1))
        tup = (X_curr,y, rows_factor, cols_factor, is_normalize_each_band, method_label_patch, i)
        tup_list.append(tup)

    
    for result in pool.starmap(HDD_HDE.calc_hdd_multiproc, tup_list):
        res,i = result
        distance_mat_arr[i] = res
        

    pool.close()  # no more tasks
    pool.join()  # wrap up current tasks


    X_curr = X[:,:,0].reshape((X.shape[0],X.shape[1],1))
    my_HDD_HDE = HDD_HDE(X_curr,y, rows_factor, cols_factor, is_normalize_each_band, method_label_patch)
    _, labels_padded, num_patches_in_row,y_patches = my_HDD_HDE.calc_hdd()
 

    print("TOTAL TIME FOR METHOD: ", time.time()-st)

    y_patches = y_patches.int()

    if torch.cuda.is_available():
        distance_mat_arr = distance_mat_arr.cpu()
        y_patches = y_patches.cpu()
        labels_padded = labels_padded.cpu()

    clf = PaviaClassifier(distance_mat_arr.numpy(), y_patches.numpy(), N_NEIGHBORS, labels_padded.numpy(), rows_factor, cols_factor, num_patches_in_row, is_divided=True, weights=weights)

    clf.classify()