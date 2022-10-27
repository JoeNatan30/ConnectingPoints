# standard
import os
import sys
from collections import Counter

# Three-party
import h5py
from sklearn.model_selection  import train_test_split
import pandas as pd

# Local

sys.path.append('../')
from utils import read_h5

class DataReader():

    def __init__(self, input_path, output_path, banned_by_distance=True, banned_by_Video_duration=True):

        classes_list, videoName_list, data_list = read_h5(input_path)

        self.df = pd.DataFrame({
            'label':classes_list,
            'video_name':videoName_list,
            'data':data_list
        })

        self.output_path = output_path
        self.banned_by_distance = banned_by_distance
        self.banned_by_Video_duration = banned_by_Video_duration


    def selectInstances(self):

        initial_df = self.df.copy()
    
        # ban some selected words
        self.df = self.df[~self.df['label'].isin(["hablar","decir","NNN","qué"])]

        set_init = set(self.df['label'])

        # Ban instance by fastDTW distance
        if self.banned_by_distance:
            bannedVideoDistDf = pd.read_csv('banned_videos_by_distance.csv', header=None)
            self.df = self.df[~self.df['video_name'].isin(bannedVideoDistDf[0])]

        # Ban instance by video duration
        if self.banned_by_Video_duration:
            bannedVideoDurDf = pd.read_csv('banned_videos_by_duration.csv', header=None)
            self.df = self.df[~self.df['video_name'].isin(bannedVideoDurDf[0])]

        # select banned videos
        #self.banDf = self.df[self.df['video_name'].isin(bannedVideoDf[0])]

        # check if there are zero instances classes after delete some instances
        set_final = set(self.df['label'])
        print(len(list(set_init)), "Glosses")
        zero_instance_list = list(set_init - set_final)

        print("Words with zero values:",zero_instance_list)

        df_to_add = initial_df[initial_df['label'].isin(zero_instance_list)]

        for unique in zero_instance_list:
            class_data = df_to_add[df_to_add["label"]==unique]
            to_add = class_data.sample(n = 1)
            self.df = pd.concat([self.df, to_add])
        
        set_final = set(self.df['label'])
        zero_instance_list = list(set_init - set_final)

        print("Words with zero values(new):",zero_instance_list)



    def saveData(self, indexOrder):

        #reorder data
        data = self.df[self.df.index.isin(list(indexOrder))]
        
        class_tmp = data['label']
        videoName_tmp = data['video_name']
        data_tmp = data['data']
       
        path = self.output_path
        print("Count:", len(indexOrder), "instances")
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

        x_pos = self.df.index

        # save the data
        self.saveData(x_pos)



################    MAIN    ################

if __name__ == "__main__":

    kpModel = "mediapipe"
    datasets = ["AEC", "PUCP_PSL_DGI156", "PUCP_PSL_DGI305"]

    dataset_out_name = [dataset if len(dataset)<6 else dataset[-6:] for dataset in datasets]
    dataset_out_name = '-'.join(dataset_out_name)

    print(f"procesing {datasets} - using {kpModel} ...\n")

    # TRAIN
    print('#'*30)
    output_path = f"../split/cleaned/{dataset_out_name}--{kpModel}-Train.hdf5"
    input_path = os.path.normpath(f'../split/{dataset_out_name}--{kpModel}-Train.hdf5')
    print(f"{input_path} --> {output_path}")
    dataReader = DataReader(input_path, output_path, 
                            banned_by_distance=True, 
                            banned_by_Video_duration=True)
    dataReader.selectInstances()
    dataReader.splitDataset()

    # VALIDATION
    print('#'*30)
    output_path = f"../split/cleaned/{dataset_out_name}--{kpModel}-Val.hdf5"
    input_path = os.path.normpath(f'../split/{dataset_out_name}--{kpModel}-Val.hdf5')
    print(f"{input_path} --> {output_path}")
    dataReader = DataReader(input_path, output_path,
                            banned_by_distance=True, 
                            banned_by_Video_duration=True)
    dataReader.selectInstances()
    dataReader.splitDataset()
