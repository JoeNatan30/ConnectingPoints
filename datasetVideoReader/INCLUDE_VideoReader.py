# Standard library imports
import os

# Third party imports
import pandas as pd
import numpy as np

# Local imports
from datasetVideoReader.util import get_list_videos

def get_data():
    
    path = "./datasets/INCLUDE/"
    video_list = get_list_videos(path,"INCLUDE")
    label = [_path.split(os.sep)[-2].split('. ')[-1] for _path in video_list['video_path']]
    # adding label column 
    video_list['label'] = label

    return video_list
