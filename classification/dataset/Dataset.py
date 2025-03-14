import torch
import numpy as np
from torch.utils.data import  Dataset
class PatientDataset(Dataset):
    def __init__(self, data_tensor, label_tensor, csv_names=None):
        self.data = data_tensor
        self.labels = label_tensor
        self.csv_names = csv_names

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.csv_names is not None:
            return self.data[idx], self.labels[idx], self.csv_names[idx]
        else:
            return self.data[idx], self.labels[idx]
