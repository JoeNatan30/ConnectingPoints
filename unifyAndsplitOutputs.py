from email import header
import os
import h5py
from collections import Counter
from sklearn.model_selection  import train_test_split
import pandas as pd
import numpy as np

from utils import read_h5

class DataReader():

    def __init__(self, datasets, kpModel, output_path):

        self.classes = []
        self.videoName = []
        self.data = []
        self.output_path = os.path.normpath(output_path)

        for dataset in datasets:

            path = os.path.normpath(f"output/{dataset}--{kpModel}.hdf5")
            classes, videoName, data = read_h5(path)
            self.classes = self.classes + classes
            self.videoName = self.videoName + videoName
            self.data = self.data + data
    

    def generate_meaning_dict(self):

        meaning = {v:k for (k,v) in enumerate(set(self.classes))}
        self.labels = [meaning[_class] for _class in self.classes]

    def fixClasses(self):

        self.classes = list(map(lambda x: x.replace('amigos', 'amigo'), self.classes))

    def selectInstances(self, selected):
    
        classes = []
        videoName = []
        data = []

        for pos in range(len(self.classes)):
            #To avoid no selected classes
            if self.classes[pos] not in selected:
                continue

            classes.append(self.classes[pos])
            videoName.append(self.videoName[pos])
            data.append(self.data[pos])

        self.classes = classes
        self.videoName = videoName
        self.data = data

    def saveData(self, indexOrder, train=True):

        #reorder data
        class_tmp = [self.classes[pos] for pos in indexOrder]
        videoName_tmp = [self.videoName[pos] for pos in indexOrder]
        data_tmp = [self.data[pos] for pos in indexOrder]
        labels_tmp = [self.labels[pos] for pos in indexOrder]
        print(set(class_tmp))
        print(len(set(class_tmp)))
        # set the path
        save_path = os.path.normpath(f"split/{self.output_path.split(os.sep)[1]}")
        save_path = save_path.split('.')

        if train:
            print("Train:", len(indexOrder))
            path = f"{save_path[0]}-Train.hdf5"
        else:
            print("Val:", len(indexOrder))
            path = f"{save_path[0]}-Val.hdf5"

        # Save H5 
        h5_file = h5py.File(path, 'w')

        for pos, (c, v, d, l) in enumerate(zip(class_tmp, videoName_tmp, data_tmp, labels_tmp)):
            grupo_name = f"{pos}"
            h5_file.create_group(grupo_name)
            h5_file[grupo_name]['video_name'] = v # video name (str)
            h5_file[grupo_name]['label'] = c # classes (str)
            h5_file[grupo_name]['data'] = d # data (Matrix)
            #h5_file[grupo_name]['class_number'] = l #label (int)
            
        h5_file.close()


    def splitDataset(self):
        
        # To know the number of instance per clases
        counter = Counter(self.classes)
        print(counter)
        # Select the words that have more or equal than 15 instances    
        counter = [word for word, count in counter.items() if count >= 15]
        print("Before ban:",len(counter))
        
        # Errase banned words
        df_banned = pd.read_csv("bannedList.csv",encoding='latin1', header=None)
        bannedList = list(df_banned[0])
        bannedList = [ban.lower() for ban in bannedList] + [ban for ban in bannedList] #+ ['lugar', 'qué?', 'sí', 'manejar', 'tú', 'ahí', 'dormir', 'cuatro', 'él', 'NNN'] #["hummm"]
        bannedList = list(set(bannedList))

        #bannedList
        selected = list(set(counter) - set(bannedList))
        print('#'*40)
        print(selected, len(selected))
        #selected = [_selected.lower() for _selected in selected]
        print("After ban:", len(selected))
        # Filter the data to have selected instances
        self.selectInstances(selected)

        # generate classes number to use it in stratified option
        self.generate_meaning_dict()
        print()
        
        print("==>",list(np.sort(np.array(selected))))
        # split the data into Train and Val (but use list position as X to reorder)
        x_pos = range(len(self.labels))
        pos_train, pos_val, y_train, y_val = train_test_split(x_pos, self.labels, train_size=0.8 , random_state=32, stratify=self.labels)
        
        # save the data
        self.saveData(pos_train,train=True)
        self.saveData(pos_val, train=False)

    
kpModel = "mediapipe"
datasets = ["AEC", "PUCP_PSL_DGI156", "PUCP_PSL_DGI305"]

dataset_out_name = [dataset if len(dataset)<6 else dataset[-6:] for dataset in datasets]
dataset_out_name = '-'.join(dataset_out_name)

print(f"procesing {datasets} - using {kpModel} ...")

output_path = f"output/{dataset_out_name}--{kpModel}.hdf5"
dataReader = DataReader(datasets, kpModel, output_path)
dataReader.fixClasses()
dataReader.splitDataset()
#splitDataset(path)

