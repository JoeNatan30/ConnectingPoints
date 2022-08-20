import os
import pandas as pd
import numpy as np
import h5py
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader

class H5Dataset(Dataset):
    def __init__(self, filePath):
        self.read_h5(filePath)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    
    def read_h5(self, path):
        
        self.data = []
        self.labels = []
        self.videoName = []

        with h5py.File(path, "r") as f:
            for index in f.keys():
                self.labels.append(f[index]['label'][...].item().decode('utf-8'))
                self.videoName.append(f[index]['video_name'][...].item().decode('utf-8'))
                self.data.append(f[index]["data"])
