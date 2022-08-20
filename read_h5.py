import os
import pandas as pd
import numpy as np
import h5py
from torchvision.io import read_image
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torch import LongTensor
from torch.nn.utils.rnn import pad_sequence


class H5Dataset(Dataset):
    def __init__(self, filePath):
        self.read_h5(filePath)
        self.generate_meaning_dict()

    def __len__(self):
        return len(self.classes)

    def __getitem__(self, idx):

        # x = T x C x K
        # y = unique int for each class (check generate_meaning_dict def)
        # class = class name
        # length = length of the timestep
        sample = {
            'x': self.data[idx],
            'y': self.getLabel(self.classes[idx]),
            'class': self.classes[idx],
            'length': self.seq_lengths[idx]
        }
        
        return sample

    def get_item(self, idx):
        return self.data[idx], self.labels[idx]

    def getLabel(self, _class):
        return self.meaning[_class]

    def generate_meaning_dict(self):
        self.meaning = {v:k for (k,v) in enumerate(set(self.classes)) }

    def sortBySeqLength(self):

        new_order = np.argsort(self.seq_lengths)

        self.seq_lengths

        self.data = [self.data[i] for i in new_order]
        self.seq_lengths = [self.seq_lengths[i] for i in new_order]
        self.classes = [self.classes[i] for i in new_order]
        self.videoName = [self.videoName[i] for i in new_order]
    
    def read_h5(self, path):
        
        self.data = []
        self.classes = []
        self.videoName = []

        with h5py.File(path, "r") as f:
            for index in f.keys():
                self.classes.append(f[index]['label'][...].item().decode('utf-8'))
                self.videoName.append(f[index]['video_name'][...].item().decode('utf-8'))
                self.data.append(torch.from_numpy(f[index]["data"][...]))


        self.seq_lengths = list(map(len, self.data))

        self.data = pad_sequence(self.data, batch_first=True)

'''
data_train = H5Dataset("output/AEC--wholepose.hdf5")
data_train_loader = DataLoader(data_train, batch_size=64, shuffle=True, )

for sample in data_train_loader:
    
    x = sample['x']
    length = sample['length']

'''