# Standard library imports
import os

# Third party imports
import pandas as pd
import numpy as np

# Local imports
from datasetVideoReader.util import get_list_videos



def get_data():
    
    path = "./datasets/AUTSL/videos/"

    # This csv contains numbers as labels
    label_train = pd.read_csv(path+'../train_labels.csv', header=None)
    label_val   = pd.read_csv(path+'../ground_truth.csv', header=None)

    meaning  = pd.read_csv(path+'../SignList_ClassId_TR_EN.csv')

    meaning_dict = {number:str_label for number, str_label in zip(meaning['ClassId'],meaning['EN'])}

    labels = pd.concat([label_train, label_val])
    labels.drop_duplicates()

    # to just process the video that are in the csv
    videoName_list = []
    label_list = []
    dataset_list = []

    for videoName, label in zip(labels[0],labels[1]):
        #print(f'{path}test/{videoName}.mp4')
        if os.path.exists(f'{path}test/{videoName}_color.mp4') or os.path.exists(f'{path}train/{videoName}_color.mp4'):
            
            if os.path.exists(f'{path}test/{videoName}_color.mp4'):
                videoName_list.append(f'{path}test/{videoName}_color.mp4')
            else:
                videoName_list.append(f'{path}train/{videoName}_color.mp4')
            
            label_list.append(meaning_dict[label])
            dataset_list.append('AUTSL')

    video_list = pd.DataFrame({
        'video_path':videoName_list,
        'label': label_list,
        'dataset': dataset_list
    })
    
    return video_list
