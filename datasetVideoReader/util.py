# Standard library imports
import os

# Third party imports
import pandas as pd

# Local imports
#

def get_list_videos(path,dataset_name):
        
    dataset_path = path

    path_list = []

    for folder_path, _ ,video_names in os.walk(dataset_path):
        
        if not video_names:
            continue
        
        for video_name in video_names:

            video_path = os.sep.join([folder_path, video_name])

            path_list.append(os.path.normpath(video_path))


    return pd.DataFrame({"video_path":path_list,
                         "dataset":[dataset_name]*len(path_list)})