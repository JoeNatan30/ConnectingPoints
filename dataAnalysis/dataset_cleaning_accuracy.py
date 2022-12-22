import sys
import os

import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics

sys.path.append('../')
from utils import create_df


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

def delete_ext(row):
    '''
    - Remove the extension of the path
    - Remove any "NewLine character" 
    '''

    row[0] = '.'.join(row[0].split('.')[:-1])
    row[0] = row[0].strip()

    return row

def is_banned_column(pd, pd_ban, column_name, matrix_ban):


    pred = df['Video name'].isin(pd_ban[0])

    '''
    Notice the following:
     - True  is set to 0 (that means that was banned)
     - False is set to 1 
    '''
    pred = pred.replace({True: 0, False: 1})

    matrix_ban['actual'] = matrix_ban['actual'] + list(df[column_name])
    matrix_ban['predicted'] = matrix_ban['predicted'] + list(pred)

    return matrix_ban

def plot_confusion_matrix(matrix_ban, name):

    title = name.replace('.png','').replace('_',' ')

    confusion_matrix = metrics.confusion_matrix(matrix_ban['actual'], matrix_ban['predicted'])
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = ['outlier', 'inlier'])
    cm_display.plot()
    cm_display.ax_.set_title(title)
    plt.savefig(name)
    plt.close()    


diff = pd.read_excel('difference.xlsx',sheet_name=None)

byVideoDur = pd.read_csv("../dataCleaningFunctions/banned_videos_by_duration.csv", header=None) 
byVideoDur = byVideoDur.apply(delete_ext,axis=1)
print(byVideoDur)

byDistance = pd.read_csv("../dataCleaningFunctions/banned_videos_by_distance.csv", header=None)
byDistance = byDistance.apply(delete_ext,axis=1)
print(byDistance)

byBothBan = byVideoDur.merge(byDistance)

matrix_by_distance = {'actual': [],
                      'predicted': []}

matrix_by_video_dur = {'actual': [],
                       'predicted': []}

matrix_by_both_ban = {'actual': [],
                      'predicted': []}

for gloss, df in diff.items():
    
    df = df.apply(video_path_format,axis=1)

    # BY DISTANCE
    matrix_by_distance = is_banned_column(df, byDistance,'TrueLabel-distance', matrix_by_distance)

    # BY VIDEO DURATION
    matrix_by_video_dur = is_banned_column(df, byVideoDur, 'TrueLabel-VidDuration', matrix_by_video_dur)
    
    # BY BOTH BAN METHOD
    matrix_by_both_ban = is_banned_column(df, byBothBan, 'TrueLabel-Strict', matrix_by_both_ban)

print(matrix_by_video_dur['actual'])
print(matrix_by_video_dur['predicted'])

plot_confusion_matrix(matrix_by_distance, 'banned_by_distance.png')
plot_confusion_matrix(matrix_by_video_dur, 'banned_by_video_duration.png')
plot_confusion_matrix(matrix_by_both_ban, 'banned_by_both_method.png')

#df_train = create_df("../split/AEC-DGI156-DGI305--mediapipe-Train.hdf5")
#df_val = create_df("../split/AEC-DGI156-DGI305--mediapipe-Val.hdf5")

