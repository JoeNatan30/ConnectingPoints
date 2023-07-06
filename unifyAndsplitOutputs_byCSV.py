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
    
    def deleteSelectedVideosToBan(self):

        df_selectedBanned = pd.read_csv("./dataCleaningFunctions/banned_selected_videos.csv", header=None)
        selectedBanned = [banned.replace('\\','/') for banned in df_selectedBanned[0]]

        # We go through the inverse of the list to use "pop" to delete the banned words
        for pos in range(len(self.videoName)-1, -1, -1):

            if self.videoName[pos] in selectedBanned:
                self.classes.pop(pos)
                self.videoName.pop(pos)
                self.data.pop(pos)

    def deleteBannedWords(self):

        df_bannedWords = pd.read_csv("./bannedList.csv",encoding='latin1', header=None)
        bannedList = list(df_bannedWords[0])

        bannedList = bannedList + [ban.lower() for ban in bannedList] + ['él','tú','','G-R']#+ ['lugar', 'qué?', 'sí', 'manejar', 'tú', 'ahí', 'dormir', 'cuatro', 'él', 'NNN'] #["hummm"]

        for pos in range(len(self.classes)-1, -1, -1):
            if self.classes[pos] in bannedList:
                self.classes.pop(pos)
                self.videoName.pop(pos)
                self.data.pop(pos)


    def generate_meaning_dict(self, words_dict):

        meaning = {v:k for (k,v) in words_dict.items()}
        self.labels = [meaning[_class] for _class in self.classes]

    def fixClasses(self):

        self.classes = list(map(lambda x: x.replace('amigos', 'amigo'), self.classes))

        _before = len(self.classes)
        self.deleteSelectedVideosToBan()

        print(f"About {_before - len(self.classes)} instances has been deleted by the ban list 'selectedVideos'")
        
        _before = len(self.classes)
        self.deleteBannedWords()

        print(f"About {_before - len(self.classes)} instances has been deleted by the ban list 'banned words'")

    def selectClasses(self, selected):
    
        for pos in range(len(self.classes)-1, -1, -1):
            if self.classes[pos] not in selected:
                self.classes.pop(pos)
                self.videoName.pop(pos)
                self.data.pop(pos)


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
        save_path = save_path.replace('$',str(len(set(class_tmp))))
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
            h5_file[grupo_name]['class_number'] = l #label (int)

        h5_file.close()


    def splitDataset(self):
        
        df_words = pd.read_csv("./incrementalList.csv",encoding='latin1', header=None)

        words = list(df_words[0])
        print(len(words),len(words),len(words),len(words),len(words))


        print('#'*40)
    
        # Filter the data to have selected instances
        self.selectClasses(words)

        # generate classes number to use it in stratified option
        self.generate_meaning_dict(df_words.to_dict()[0])

        # split the data into Train and Val (but use list position as X to reorder)
        x_pos = range(len(self.labels))
        pos_train, pos_val, y_train, y_val = train_test_split(x_pos, self.labels, train_size=0.8 , random_state=32, stratify=self.labels)
        
        # save the data
        self.saveData(pos_train,train=True)
        self.saveData(pos_val, train=False)

    
kpModel = "mediapipe"
datasets = ["PUCP_PSL_DGI305"] #["AEC", "PUCP_PSL_DGI156", "PUCP_PSL_DGI305", "WLASL", "AUTSL"]

dataset_out_name = [dataset if len(dataset)<6 else dataset[-6:] for dataset in datasets]
dataset_out_name = '-'.join(dataset_out_name)

print(f"procesing {datasets} - using {kpModel} ...")

output_path = f"output/{dataset_out_name}--$--incremental--{kpModel}.hdf5"

dataReader = DataReader(datasets, kpModel, output_path)
dataReader.fixClasses()

dataReader.splitDataset()
#splitDataset(path)

