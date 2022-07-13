# Standard library imports
import os

# Third party imports
import pandas as pd
import numpy as np

# Local imports
from util import get_list_videos

def get_data():
    
    path = "./datasets/AEC/Videos/SEGMENTED_SIGN/"
    video_list = get_list_videos(path,"AEC")

    # .split('/')[-1] to take the video name of the path
    # .split('_')[:-1] to take the label of the video
    label = [_path.split('/')[-1].split('_')[:-1] for _path in video_list['video path']]

    # to change '' label to PeruSil standar '???'
    label = [_label[0] if _label != '' else '???' for _label in label ]

    # adding label column 
    video_list['label'] = label

    return video_list

get_data()