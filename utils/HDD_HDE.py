import numpy as np
import torch
from consts import CONST_K,ALPHA,TOL,CONST_C, N_NEIGHBORS, APPLY_2_NORM, dist_dtype
import time


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cpu = torch.device("cpu")


class HDD_HDE:
    def __init__(self, X,y, rows_factor, cols_factor, is_normalize_each_band=True, method_label_patch='center'):
        self.X = X
        self.y = y
        self.rows_factor = rows_factor
        self.cols_factor = cols_factor
        self.is_normalize_each_band = is_normalize_each_band
        self.method_label_patch = method_label_patch


 
    def hdd(X,P):
        d_HDD = torch.zeros_like(P, device=device)

        for k in range(CONST_K + 1):
            norms = torch.cdist(X[k], X[k])
            sum_matrix = 2 * torch.arcsinh((2 ** (-k * ALPHA + 1)) * norms)
            d_HDD += sum_matrix

            del norms
            del sum_matrix

        return d_HDD


    
    def svd_symmetric(M):
        s,u = torch.linalg.eigh(M)

        s, indices = torch.sort(s, descending=True)
        
        u = u[:, indices]

        v = u.clone()
        v[:,s<0] = -u[:,s<0] #replacing the corresponding columns with negative sign

        torch.abs(s, out=s)

        torch.where(s>TOL, s, torch.tensor([TOL], device=device), out=s)

        return u, s, torch.t(v)

        
    def calc_svd_p(d):
        epsilon = CONST_C*torch.mean(d, dim=tuple(np.arange(len(d.shape))))

        W = torch.exp(-1*d/epsilon)

        S_vec = torch.sum(W,dim=1)
        S = torch.diag(1/S_vec)
        
        del S_vec

        S = S.type(torch.float32)
        W = W.type(torch.float32)

        W_gal_tmp = torch.mm(S,W)
        W_gal_tmp = W_gal_tmp.type(torch.float32)

        W_gal = torch.mm(W_gal_tmp,S)
        
        
        del W_gal_tmp
        del W
        del S

        D_vec = torch.sum(W_gal,dim=1)
        
        D_minus_half = torch.diag(1 / torch.sqrt(D_vec))
        D_plus_half = torch.diag(torch.sqrt(D_vec))
        del D_vec
        
        M = torch.matmul(torch.matmul(D_minus_half,W_gal),D_minus_half)
        
        del W_gal

        U,S,UT = HDD_HDE.svd_symmetric(M)
        del M
        
        res = (torch.matmul(D_minus_half,U)),(S),(torch.matmul(UT,D_plus_half))

        del S
        del D_minus_half
        del D_plus_half
        del U
        del UT

        return res


 
    def hde(shortest_paths_mat):
        #   print("hde_torch", flush=True)
        U, S_keep, Vt = HDD_HDE.calc_svd_p(shortest_paths_mat)
        
        U = U.double()
        S_keep = S_keep.double()
        Vt = Vt.double()

        #   print("hde_torch- BEFORE LOOP", flush=True)
        X = torch.zeros((CONST_K + 1, shortest_paths_mat.shape[0], shortest_paths_mat.shape[1] + 1), dtype=dist_dtype, device=device)
        for k in range (0, CONST_K + 1):
            S = torch.float_power(S_keep, 2 ** (-k))

            aux = torch.matmul(torch.matmul(U,torch.diag(S)),Vt)

            aux = torch.t(torch.sqrt((torch.where(aux > TOL, aux, torch.tensor([TOL], device=device)))))
            X[k] = torch.t(torch.cat((aux, torch.reshape(torch.full((shortest_paths_mat.shape[0],), 2 ** (k * ALPHA - 2), device=device),(1, -1))), dim=0))

            del aux
            del S
        
        #   print("hde_torch- AFTER LOOP", flush=True)

        del U
        del Vt
        del S_keep

        return X


 
    def padWithZeros(X, left_margin, right_margin, top_margin, bottom_margin, dim=3):
        if dim == 3:
            newX = torch.zeros((X.shape[0] + left_margin + right_margin, X.shape[1] + top_margin + bottom_margin, X.shape[2]), dtype=X.dtype, device=device)
            newX[left_margin:X.shape[0] + left_margin, top_margin:X.shape[1] + top_margin, :] = X
        
        elif dim == 2:
            newX = torch.zeros((X.shape[0] + left_margin + right_margin, X.shape[1] + top_margin + bottom_margin), dtype=X.dtype, device=device)
            newX[left_margin:X.shape[0] + left_margin, top_margin:X.shape[1] + top_margin] = X

        else:
            newX = []

        return newX 

 
    def calc_patch_label(labels, i, j, rows_factor, cols_factor, method):
        if method=='center':
            return labels[i*rows_factor + rows_factor//2, j*cols_factor + cols_factor//2]
        elif method=='most_common':
            labels_patch = (labels[i*rows_factor : (i+1)*rows_factor, j*cols_factor : (j+1)*cols_factor]).int()
            counts = torch.bincount(labels_patch.flatten())

            # in order to not let 0 values take over and set many labels to 0 which leads to small number of non zero labeled patches
            counts[0]=1

            return torch.argmax(counts)
        
        print("ERROR- INCORRECT METHOD FOR LABELING PATCHES")

 
    def patch_data(self, data):
        rows, cols, channels = data.shape

        left_margin = ((-rows) % self.rows_factor) // 2
        right_margin = ((-rows) % self.rows_factor + 1) // 2
        top_margin = ((-cols) % self.cols_factor) // 2
        bottom_margin = ((-cols) % self.cols_factor + 1) // 2

        data = HDD_HDE.padWithZeros(data, left_margin=left_margin, right_margin=right_margin, top_margin=top_margin, bottom_margin=bottom_margin)
        labels = HDD_HDE.padWithZeros(self.y, left_margin=left_margin, right_margin=right_margin, top_margin=top_margin, bottom_margin=bottom_margin, dim=2)


        new_rows, new_cols, _ = data.shape

        patched_data = torch.empty((new_rows // self.rows_factor, new_cols // self.cols_factor, self.rows_factor, self.cols_factor, channels), dtype=data.dtype, device=device)
        patched_labels = torch.zeros((patched_data.shape[0], patched_data.shape[1]), dtype=labels.dtype, device=device)

        for i in range(new_rows // self.rows_factor):
            for j in range(new_cols // self.cols_factor):
                datapoint = data[i*self.rows_factor: (i+1)*self.rows_factor, j*self.cols_factor: (j+1)*self.cols_factor, :]
                patched_data[i, j] = datapoint
                patched_labels[i, j] = HDD_HDE.calc_patch_label(labels, i, j, self.rows_factor, self.cols_factor, method=self.method_label_patch)

        return patched_data, patched_labels, labels

 
    def normalize_each_band(X):
        
        X_normalized = torch.zeros_like(X, dtype=dist_dtype, device=device)

        for i in range(X.shape[2]):
            X_band = X[:,:,i]
            scaled_data = torch.div((torch.sub(X_band,torch.min(X_band).item())),((torch.max(X_band)).item() - (torch.min(X_band)).item()))
            X_normalized[:,:,i] = scaled_data

        
        return X_normalized

 
    def calc_P(d, apply_2_norm=APPLY_2_NORM):
        epsilon = CONST_C*torch.mean(d, dim=tuple(np.arange(len(d.shape))))

        # print("epsilon: ", epsilon, flush=True)
        W = torch.exp(-1*d/epsilon)

        if apply_2_norm:
            S_vec = torch.sum(W,dim=1)

            S = torch.diag(1/S_vec)
            del S_vec
            
            W_gal = torch.matmul(torch.matmul(S,W),S)
            del S

            D_vec = torch.sum(W_gal,dim=1)
            D = torch.diag(1 / D_vec)
            del D_vec
            P = torch.matmul(D,W_gal)

            del W_gal
            del D

        else:
            D_vec = torch.sum(W,dim=1)
            D = torch.diag(1/ D_vec)
            del D_vec
            P = torch.matmul(D,W)

            del D

        del W

        return P

    
    def prepare(self):
        X = self.X
        
        if self.is_normalize_each_band:
            X = HDD_HDE.normalize_each_band(X)

        X_patches, y_patches, labels_padded= self.patch_data(X)

        num_patches_in_row = y_patches.shape[1]

        y_patches = y_patches.flatten()

        X_patches = torch.reshape(X_patches, (-1, np.prod(X_patches.shape[2:])))


        distances = torch.cdist(X_patches, X_patches)
        
        del X_patches

        return distances,y_patches,num_patches_in_row, labels_padded 


 
    def calc_hdd(self):
        distances,y_patches,num_patches_in_row, labels_padded = self.prepare()
        
        hdd_mat = HDD_HDE.run_method(distances)

        return hdd_mat, labels_padded, num_patches_in_row,y_patches

    def calc_hdd_multiproc(X_curr,y, rows_factor, cols_factor, is_normalize_each_band, method_label_patch, i):
        my_hdd_hde = HDD_HDE(X_curr,y, rows_factor, cols_factor, is_normalize_each_band, method_label_patch)
        hdd_mat, labels_padded, num_patches_in_row,y_patches = my_hdd_hde.calc_hdd()
        return hdd_mat, i

    def run_method(distances):
        P = HDD_HDE.calc_P(distances, apply_2_norm=APPLY_2_NORM)
        HDE = HDD_HDE.hde(distances)
        del distances
        torch.abs(HDE, out=HDE)

        hdd_mat = HDD_HDE.hdd(HDE, P)

        del HDE

        return hdd_mat


