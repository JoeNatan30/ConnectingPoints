import pandas as pd
import os
import cv2

def fixPath(row):
    '''
    Some video names of the dataset have its name changed (probably happended during unzip process)
    This def fix all '_' that had been changed for '?'
    
    Note: you have to modify this def if you have paths that have '?' and '_' as part of its name
    '''
    row['comp'] = row['comp'].replace('?','_')
    #row['full'] = row['full'].replace('?','_')

    return row

list_path = []
for folder, _, files in os.walk(os.path.normpath("../datasets/")):
    files = [_file for _file in files if _file[-3:]=='mp4']
    for file in files:
        '''
        we got to values from here:
        - comp: original_video_name/short_video_name
        - full: the path of a short video from the folder where this script is runned  
        '''
        comp = '\\'.join([folder.split(os.sep)[-1],file])
        origin_path = os.sep.join([folder,file])

        list_path.append((comp, origin_path))

df_paths = pd.DataFrame(list_path,columns=["comp","full"])
df_paths_fixed = df_paths.apply(fixPath, axis=1)

def get_length(row):
    '''
    To get the duration (seconds) of a short video
    '''
    result = df_paths[df_paths["comp"] == row]

    if len(result)==0:
        result = df_paths_fixed[df_paths_fixed["comp"] == row]

    assert len(result) != 0 , f'check if the video {row} , is in the dataset file'
    assert len(result) == 1 , f'check if the video {row} is repeated in the dataset file'


    if len(result) != 1:
        print(result)

    cap = cv2.VideoCapture(result['full'].values[0])

    assert cap.isOpened(), f"Unable to read camera feed {result['full'].values[0]}"       
    
    fps = cap.get(cv2.CAP_PROP_FPS) 
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    duration = frame_count/fps

    return duration

general_list = pd.read_csv("results.csv")
print(general_list)
video_list = []

#unique_glosses = list(set(general_list["classes"]))

general_list['length'] = general_list['videoName'].apply(get_length)

pMayor = general_list.groupby("classes")["length"].quantile(0.9)
pMinus = general_list.groupby("classes")["length"].quantile(0.1)

# Crea un diccionario con los valores de percentil para cada categoría
percentiles_minus = pMinus.to_dict()
percentiles_mayor = pMayor.to_dict()

print(percentiles_minus)

to_drop = []

# Recorre cada fila del dataframe
for i, row in general_list.iterrows():

    # Si el valor de la fila está por debajo (o encima) del percentil de su categoría, bans the instance by adding thir index in the list
    if row["length"] < percentiles_mayor[row["classes"]] and row["length"] > percentiles_minus[row["classes"]] :
        to_drop.append(i)

# Elimina las filas de la lista
general_list = general_list.drop(to_drop)

ban_list = []

pd_ban = pd.DataFrame(general_list['videoName'])
print("banned")
print(pd_ban)
pd_ban.to_csv("banned_videos_by_duration.csv", header=False, index=False)


'''

video_df = pd.DataFrame(video_list, columns=['video_path','video_name'])

ban_list = []

pd_ban = pd.DataFrame(ban_list)
print(pd_ban)
pd_ban.to_csv("banned_videos_by_duration.csv", header=False, index=False)


import sys

import pandas as pd
import matplotlib.pyplot as plt

sys.path.append('../')
from utils import create_df

def get_length(row):
    return len(row)


df_train = create_df("../split/AEC-DGI156-DGI305--mediapipe-Train.hdf5")
df_val = create_df("../split/AEC-DGI156-DGI305--mediapipe-Val.hdf5")

df_train['length'] = df_train['data'].apply(get_length)
df_val['length'] = df_val['data'].apply(get_length)

df_train.boxplot(column=['length'], 
                 by="classes",
                 #title="Training- Distribution by class",
                 rot=90,
                 figsize=(15, 8)).get_figure().savefig('3-dataset-length-training.png')


df_val.boxplot(column=['length'], 
                 by="classes",
                 #title="Training- Distribution by class",
                 rot=90,
                 figsize=(15, 8)).get_figure().savefig('3-dataset-length-validation.png')

print("Training:  ",df_train['length'].mean())
print("Validation:",df_val['length'].mean())

'''