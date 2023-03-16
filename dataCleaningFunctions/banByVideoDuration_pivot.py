import pandas as pd
import os
import cv2

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

    #assert len(result) != 0 , f'check if the video {row} , is in the dataset file'
    #assert len(result) == 1 , f'check if the video {row} is repeated in the dataset file'


    if len(result) != 1:
        return 0
        print(result)

    cap = cv2.VideoCapture(result['full'].values[0])

    assert cap.isOpened(), f"Unable to read camera feed {result['full'].values[0]}"       
    
    fps = cap.get(cv2.CAP_PROP_FPS) 
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    duration = frame_count/fps

    return duration

general_list = pd.read_csv("results.csv")

general_list['length'] = general_list['videoName'].apply(get_length)

ban_list = []

for name, length in zip(general_list['videoName'] ,general_list['length']):

    if length <= 0.26:
        ban_list.append(name)


ban_df = pd.DataFrame(ban_list)
print(ban_df)
ban_df.to_csv("banned_videos_by_duration.csv", header=False, index=False)