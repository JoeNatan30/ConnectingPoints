from email import header
import os
import h5py
from collections import Counter
from sklearn.model_selection  import train_test_split
import pandas as pd

from utils import read_h5

class DataReader():

    def __init__(self, dataset,  kpModel, output_path):

        classes_list  = []
        videoName_list = []
        data_list = []

        classes_train, videoName_train, data_train = read_h5(f'split/{dataset}--{kpModel}-Train.hdf5')
        classes_val, videoName_val, data_val = read_h5(f'split/{dataset}--{kpModel}-Val.hdf5')

        classes_list = classes_train + classes_val
        videoName_list = videoName_train + videoName_val
        data_list = data_train + data_val

        self.output_path = os.path.normpath(output_path)

        self.df = pd.DataFrame({
            'label':classes_list,
            'video_name':videoName_list,
            'data':data_list
        })
    
    def generate_meaning_dict(self):

        meaning = {v:k for (k,v) in enumerate(set(self.df['label']))}
        self.labels = [meaning[_class] for _class in self.df['label']]

    def selectInstances(self):
    
        # To know the number of instance per clases
        counter = self.df.groupby('label').size()

        df_banned = pd.read_csv("bannedList.csv",encoding='latin1', header=None)
        bannedList = list(df_banned[0])

        print("Before ban:",len(counter))

        selectedGloss = pd.read_csv('selected_gloss.csv', header=None)

        # chosse banned videos
        #self.banDf = self.df[self.df['label'].isin(selectedGloss[0])]

        # chosse not banned videos and filter by bannedList again
        self.df = self.df[self.df['label'].isin(selectedGloss[0])]
        self.df = self.df[~self.df['label'].isin(bannedList)]

        counter = self.df.groupby('label').size()
        print("After ban:",len(counter))

    def saveData(self, indexOrder, train=True):

        #reorder data
        data = self.df[self.df.index.isin(list(indexOrder))]
        
        class_tmp = data['label']
        videoName_tmp = data['video_name']
        data_tmp = data['data']

        # set the path
        save_path = os.path.normpath(self.output_path)
        save_path = save_path.split('.')

        if train:
            print("Train:", len(indexOrder))
            path = f"{save_path[0]}-Train-short.hdf5"
        else:
            print("Val:", len(indexOrder))
            path = f"{save_path[0]}-Val-short.hdf5"

        # Save H5

        h5_file = h5py.File(path, 'w')

        for pos, (c, v, d) in enumerate(zip(class_tmp, videoName_tmp, data_tmp)):
            grupo_name = f"{pos}"
            h5_file.create_group(grupo_name)
            h5_file[grupo_name]['video_name'] = v # video name (str)
            h5_file[grupo_name]['label'] = c # classes (str)
            h5_file[grupo_name]['data'] = d # data (Matrix)
            #h5_file[grupo_name]['class_number'] = l #label (int)
            
        h5_file.close()


    def splitDataset(self):

        self.generate_meaning_dict()

        # split the data into Train and Val (but use list position as X to reorder)
        x_pos = self.df.index

        pos_train, pos_val, y_train, y_val = train_test_split(x_pos, self.df['label'], train_size=0.8 , random_state=32, stratify=self.df['label'])

        # save the data
        self.saveData(pos_train,train=True)
        self.saveData(pos_val, train=False)
    
kpModel = "mediapipe"
datasets = ["AEC", "PUCP_PSL_DGI156", "PUCP_PSL_DGI305"]



dataset_out_name = [dataset if len(dataset)<6 else dataset[-6:] for dataset in datasets]
dataset_out_name = '-'.join(dataset_out_name)

print(f"procesing {datasets} - using {kpModel} ...")

output_path = f"split/cleaned/{dataset_out_name}--{kpModel}-.hdf5"
print(output_path)

dataReader = DataReader(dataset_out_name, kpModel, output_path)
dataReader.selectInstances()
dataReader.splitDataset()
'''

#splitDataset(path)
'''
