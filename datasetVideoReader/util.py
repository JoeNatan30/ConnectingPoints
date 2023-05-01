# Standard library imports
import os

# Third party imports
import pandas as pd

# Local imports
#
video_ext = ['3gp', 'mpg', 'mpeg', 'mp4', 'm4v', 
             'm4p', 'ogv', 'ogg', 'mov', 'webm']
video_ext = video_ext + [_ext.upper() for _ext in video_ext]

def get_list_videos(path,dataset_name):
        
    dataset_path = path

    path_list = []

    for folder_path, _ ,video_names in os.walk(dataset_path):
        
        if not video_names:
            continue
        
        for video_name in video_names:

            ext = video_name.split('.')[-1]
            if ext in video_ext:
                video_path = os.sep.join([folder_path, video_name])
                path_list.append(os.path.normpath(video_path))


    return pd.DataFrame({"video_path":path_list,
                         "dataset":[dataset_name]*len(path_list)})