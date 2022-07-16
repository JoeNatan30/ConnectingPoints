# Standard library imports

# Third party imports
import pandas as pd

# Local imports
from datasetVideoReader.util import get_list_videos

def simplyfy_json_dict(json_dict):
    
    id_temp = []
    label_temp =[]

    for pos, word in enumerate(json_dict["gloss"]):
            instances = json_dict["instances"][pos]
            for inst in instances:
                id_temp.append(inst['video_id'])
                label_temp.append(word)
    
    return pd.DataFrame({'video_id': id_temp,
                         'label':label_temp})

def get_data():
    
    path = "./datasets/WLASL/start_kit/videos"
    video_list = get_list_videos(path,"WLASL")

    video_list['video_id'] = video_list['video_path'].apply(lambda x: x.split('/')[-1].split('.')[0])
    #video_list['video_id'] = [path.split('/')[-1].split('.')[0] for path in video_list['video_path']]

    json_dict = pd.read_json("./datasets/WLASL/start_kit/WLASL_v0.3.json")
    
    json_simply = simplyfy_json_dict(json_dict)

    video_list = pd.merge(video_list, json_simply,on='video_id')

    #del video_list["video_id"]
    return video_list

