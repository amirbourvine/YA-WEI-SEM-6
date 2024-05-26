import numpy as np
from Classifier import * 

class PaviaClassifier(Classifier):

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

            return distances_mat_new, labels_new, new_res

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
