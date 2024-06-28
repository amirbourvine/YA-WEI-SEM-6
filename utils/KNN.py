import numpy as np

class kNN:
    def __init__(self, n_neighbors, is_divided):
        # assume non-negative labels
        self.n_neighbors = n_neighbors
        self.is_divided = is_divided

    def fit(self, labels, patch_to_points_dict):
      self.labels = labels
      self.patch_to_points_dict = patch_to_points_dict

      #   print("patch_to_points_dict: ", patch_to_points_dict)
      #   print("labels: ", labels)
      return self

    def score(self, distances, indices_test, y, weights):
        if self.is_divided:
            return self.score_divided(distances, indices_test, y, weights)
        return self.score_not_devided(distances, indices_test, y)


    def score_not_devided(self, distances, indices_test, y):
        print(f"distances.shape: {distances.shape}", flush=True)
        
        y_mat = np.tile(self.labels,(distances.shape[0],1))
        
        print(f"y_mat.shape: {y_mat.shape}", flush=True)
        
        sorted_distances_labels = np.take_along_axis(y_mat, np.argsort(distances, axis=1), axis=1)

        X_kNN = sorted_distances_labels[:, :self.n_neighbors]

        predictions = np.zeros(X_kNN.shape[0])
        for i in range(predictions.shape[0]):
            curr = np.bincount(X_kNN[i,:])
            predictions[i] = np.argmax(curr)

        total_preds = 0
        total_correct = 0
        preds = []
        gt = []

        for ind in range(predictions.shape[0]):
            ind_patch = indices_test[ind]
            i_start,i_end,j_start,j_end = self.patch_to_points_dict[ind_patch]
            for i in range(i_start,i_end):
                for j in range(j_start,j_end):
                    if y[i,j]!=0:
                        total_preds += 1
                        if y[i,j] == predictions[ind]:
                            total_correct += 1

                        preds.append(predictions[ind])
                        gt.append(y[i,j])
        
        print("total classified: ", total_preds)

        return total_correct/total_preds, preds,gt
    
    def score_divided(self, distances_arr, indices_test, y, weights):
        y_mat = np.tile(self.labels,(distances_arr[0].shape[0],1))

        sorted_distances_labels = np.ndarray(shape=(distances_arr.shape[0],distances_arr[0].shape[0],distances_arr[0].shape[1]))
        for i in range(distances_arr.shape[0]):
            sorted_distances_labels[i,:,:] = np.take_along_axis(y_mat, np.argsort(distances_arr[i], axis=1), axis=1)

        X_kNN = sorted_distances_labels[:,:, :self.n_neighbors]

        new_weights = None
        if weights is not None:
            new_weights = np.repeat(weights, X_kNN.shape[-1])


        X_kNN = np.swapaxes(X_kNN,0,1)

        X_kNN = X_kNN.reshape((X_kNN.shape[0],X_kNN.shape[1]*X_kNN.shape[2]))
        X_kNN = (X_kNN).astype(int)
    

        predictions = np.zeros(X_kNN.shape[0])
        for i in range(predictions.shape[0]):
            # print("X_kNN[i,:]: ", X_kNN[i,:].shape)
            # print("new_weights: ", new_weights.shape)
            curr = np.bincount(X_kNN[i,:], weights=new_weights)
            predictions[i] = np.argmax(curr)

        total_preds = 0
        total_correct = 0
        preds = []
        gt = []
        for ind in range(predictions.shape[0]):
            ind_patch = indices_test[ind]
            i_start,i_end,j_start,j_end = self.patch_to_points_dict[ind_patch]
            for i in range(i_start,i_end):
                for j in range(j_start,j_end):
                    if y[i,j]!=0:
                        total_preds += 1
                        if y[i,j] == predictions[ind]:
                            total_correct += 1

                        preds.append(predictions[ind])
                        gt.append(y[i,j])
        
        return total_correct/total_preds, preds,gt