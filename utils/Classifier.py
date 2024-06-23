import numpy as np
import random
import time
from KNN import kNN
import consts

class Classifier:
    def __init__(self, distances_mat, labels, n_neighbors, labels_padded, rows_factor, cols_factor, num_patches_in_row, is_divided=True, weights=None, random_seed=None):
        self.distances_mat = distances_mat
        self.labels = labels
        self.n_neighbors = n_neighbors
        self.labels_padded = labels_padded
        self.rows_factor = rows_factor
        self.cols_factor = cols_factor
        self.num_patches_in_row = num_patches_in_row
        self.is_divided = is_divided
        self.test_size = consts.TEST_SIZE
        self.weights = weights
        self.random_seed = random_seed

    def split_train_test(self, distances_mat, labels):
        """
        labels is np.array
        returns the train-test split:
        for train- labels and distances is train_numXtrain_num mat- distances between all the training points
        for test- distances is test_numXtrain_num mat- distances between each test point to all of the train points
        """
        num_training = int(labels.shape[0]*(1-self.test_size))
        if self.random_seed is not None:
            random.seed(self.random_seed)
        indices_train = random.sample(range(0, labels.shape[0]), num_training)
        indices_train = np.sort(indices_train)
        # print(indices_train)
        if not self.is_divided:
            dmat_train = distances_mat[np.ix_(indices_train, indices_train)]
        else:
            dmat_train = np.ndarray(shape=(distances_mat.shape[0],), dtype=np.ndarray)
            for i in range(dmat_train.shape[0]):
                dmat_train[i] = (distances_mat[i])[np.ix_(indices_train, indices_train)]
        
        labels_train = labels[np.ix_(indices_train)]

        indices_test = []
        for i in range(labels.shape[0]):
            if i in indices_train:
                continue
            indices_test.append(i)
        
        indices_test = np.array(indices_test)
        # print(indices_test)

        if not self.is_divided:
            dmat_test = distances_mat[np.ix_(indices_test, indices_train)]
        else:
            dmat_test = np.ndarray(shape=(distances_mat.shape[0],), dtype=np.ndarray)
            for i in range(dmat_test.shape[0]):
                dmat_test[i] = (distances_mat[i])[np.ix_(indices_test, indices_train)]

        labels_test = labels[np.ix_(indices_test)]

        return indices_train,dmat_train,labels_train,indices_test,dmat_test,labels_test

    def patch_to_points(self):
        """
        create a dict where key is i- index of patch in labels and value is (i_start, i_end, j_start, j_end)
        which are the boundaries of indices of points of this patch
        """
        res = {}
        for i in range(self.labels.shape[0]):
            i_patch = i // self.num_patches_in_row
            j_patch = i % self.num_patches_in_row

            i_start = i_patch*self.rows_factor
            j_start = j_patch*self.cols_factor
            res[i] = (i_start, i_start+self.rows_factor, j_start, j_start+self.cols_factor)

        return res


    def classify(self):
        raise NotImplemented
