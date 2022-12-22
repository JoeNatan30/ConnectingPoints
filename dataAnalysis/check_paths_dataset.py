import sys

import pandas as pd

sys.path.append('../')
from utils import read_h5

def video_path_format(row):
    '''
    - Delete any "NewLine Character" using .strip() 
    - the first '-' in the path have to be replaced by '\\' 
    - to make it comparable with the path of the banned videos
    '''
    
    video_name = row['Video name']

    video_name = video_name.strip()
    video_name = video_name.split('-')
    video_name = video_name[0] + '\\' + '-'.join(video_name[1:])

    row['Video name'] = video_name

    return row

diff = pd.read_excel('difference.xlsx',sheet_name=None)

diff_list = []

for gloss, df in diff.items():
    df = df.apply(video_path_format,axis=1)
    df['classes'] = gloss
    diff_list.append(df)

df_complete = pd.concat(diff_list, ignore_index=True)

glosses, videoName, data = read_h5("../split/AEC-DGI156-DGI305--mediapipe-Val.hdf5")

df_val = pd.DataFrame.from_dict({
    "classes":glosses,
    "videoName": videoName,
    "data":data,  
})

glosses, videoName, data = read_h5("../split/AEC-DGI156-DGI305--mediapipe-Train.hdf5")

df_train = pd.DataFrame.from_dict({
    "classes":glosses,
    "videoName": videoName,
    "data":data,  
})


df_dataset = pd.concat([df_train, df_val], ignore_index=True)


is_in = df_dataset['videoName'].isin(df_dataset['videoName'])
print(is_in, is_in.all())
is_in = df_dataset[df_dataset['videoName'].isin(df_complete['Video name'])]
