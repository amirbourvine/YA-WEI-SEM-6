import pandas as pd
import os

class datasetLoader:
    def __init__(self, csv_path, gt_csv_path=None):
        self.df = pd.read_csv(csv_path)
        self.df.drop(self.df.columns[0], axis=1, inplace = True)

        if gt_csv_path!=None:
            self.gt = pd.read_csv(gt_csv_path)
            self.gt.drop(self.gt.columns[0], axis=1, inplace = True)
    
    def read_dataset(self, gt=False):
        if gt:
            return self.gt
        return self.df
        