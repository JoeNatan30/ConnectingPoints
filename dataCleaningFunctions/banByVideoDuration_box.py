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

groupby_class= general_list.groupby("classes")["length"]

bound_list = []
for gloss, df in groupby_class:
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3-Q1
    lowqe_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    bound_list.append((gloss, lowqe_bound, upper_bound))

bound_df = pd.DataFrame(bound_list, columns=['classes', 'lower_bound', 'upper_bound'])

# Crea un diccionario con los valores de percentil para cada categoría

to_drop = []

for i, row in general_list.iterrows():

    limits = bound_df[bound_df['classes'] == row['classes']]

    length = float(row["length"])
    lower = float(limits['lower_bound'])
    upper = float(limits['upper_bound'])

    if  length >= lower and length <= upper:
        to_drop.append(i)

# Elimina las filas de la lista
general_list = general_list.drop(to_drop)

ban_list = []

pd_ban = pd.DataFrame(general_list['videoName'])
print('#'*50)
print("BANNED")
print(pd_ban)
pd_ban.to_csv("banned_videos_by_duration.csv", header=False, index=False)
