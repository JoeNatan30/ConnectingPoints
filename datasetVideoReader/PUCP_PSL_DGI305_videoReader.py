# Standard library imports
import os

# Third party imports
import pandas as pd

# Local imports
from datasetVideoReader.util import get_list_videos

def get_data():
    
    path = "./datasets/PUCP_PSL_DGI305/Videos/SEGMENTED_SIGN"
    video_list = get_list_videos(path,"PUCP_PSL_DGI305")

    # .split('/')[-1] to take the video name of the path
    # .split('_')[:-1] to take the label of the video
    labels = [_path.split(os.sep)[-1].split('_')[:-1]for _path in video_list['video_path']]

    labels = [_label[0] for _label in labels]

    # JUST TO MIX VARIANTS
    #labels = [_label[0].split("-")[0] if len(_label[0].split("-")[0]) > 1 \
    #                                  else _label[0]  \
    #          for _label in labels]
    # to change '' label to PeruSil standar '???'
    print(labels)
    labels = [_label if _label != '' else '???' for _label in labels ]

    labels = [_label if '-' in _label else _label.lower() for _label in labels ]
    
    # adding label column 
    video_list['label'] = labels

    return video_list
