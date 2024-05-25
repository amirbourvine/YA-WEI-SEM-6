import numpy as np
import random
import time
from KNN import kNN
from consts import TEST_SIZE

class Classifier:
    def __init__(self, distances_mat, labels, n_neighbors, labels_padded, rows_factor, cols_factor, num_patches_in_row, is_divided, weights=None):
        self.distances_mat = distances_mat
        self.labels = labels
        self.n_neighbors = n_neighbors
        self.labels_padded = labels_padded
        self.rows_factor = rows_factor
        self.cols_factor = cols_factor
        self.num_patches_in_row = num_patches_in_row
        self.is_divided = is_divided
        self.test_size = TEST_SIZE
        self.weights = weights


    def throw_0_labels(self, patch_to_points_dict):
        non_zero_indices = np.where(self.labels != 0)[0]
        non_zero_indices = np.sort(non_zero_indices)

        new_res = {}
        for i in range(non_zero_indices.shape[0]):
            new_res[i] = patch_to_points_dict[non_zero_indices[i]]

        labels_new = self.labels[np.ix_(non_zero_indices)]

        if not self.is_divided:
            dmat_new = self.distances_mat[np.ix_(non_zero_indices, non_zero_indices)]
            return dmat_new,labels_new, new_res
        else:
            distances_mat_new = np.ndarray(shape=(self.distances_mat.shape[0],), dtype=np.ndarray)
            for i in range(distances_mat_new.shape[0]):
                distances_mat_new[i] = (self.distances_mat[i])[np.ix_(non_zero_indices, non_zero_indices)]

            return distances_mat_new,labels_new, new_res


    def split_train_test(self, distances_mat, labels):
        """
        labels is np.array
        returns the train-test split:
        for train- labels and distances is train_numXtrain_num mat- distances between all the training points
        for test- distances is test_numXtrain_num mat- distances between each test point to all of the train points
        """
        num_training = int(labels.shape[0]*(1-self.test_size))
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
        patch_to_points_dict = self.patch_to_points()

        distances_mat, labels,patch_to_points_dict = self.throw_0_labels(patch_to_points_dict)

        indices_train,dmat_train,labels_train,indices_test,dmat_test,labels_test = self.split_train_test(distances_mat, labels)

        clf = kNN(n_neighbors=self.n_neighbors, is_divided=self.is_divided)
        clf.fit(labels=labels_train, patch_to_points_dict=patch_to_points_dict)


        train_acc, train_preds,train_gt = clf.score(dmat_train, indices_train, self.labels_padded, self.weights)
        test_acc, test_preds,test_gt= clf.score(dmat_test, indices_test, self.labels_padded, self.weights)

        print("Train Accuracy: ",train_acc)
        print("Test Accuracy: ",test_acc)

        return train_acc,test_acc, test_preds,test_gt
