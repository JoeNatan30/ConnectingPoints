# Standard library imports
import os

# Third party imports
import pandas as pd

# Local imports
from datasetVideoReader.util import get_list_videos


def get_data():

    path = "./datasets/WLASL/rgb/WLASL2000"

    video_list = get_list_videos(path,"WLASL")

    video_list['video_id'] = video_list['video_path'].apply(lambda x: x.split(os.sep)[-1].split('.')[0])

    csv_path = "./datasets/WLASL/videos_info.csv"
    csv_data =  pd.read_csv(csv_path, dtype={'id': str})
    csv_data = csv_data.rename(columns={'id': "video_id"})
    csv_data = csv_data.rename(columns={'label': "id"})
    csv_data = csv_data.rename(columns={'gloss': "label"})

    video_list = pd.merge(video_list, csv_data, on='video_id', )

    del(video_list['relative_path'])

    video_list = video_list[video_list['group']==2000]

    del(video_list['group'])

    return video_list
    '''
    This commented code is part is used for the wlasl obtanied from a website that is not kaggle
    '''
    # def simplyfy_json_dict(json_dict):
    
    # id_temp = []
    # label_temp =[]

    # for pos, word in enumerate(json_dict["gloss"]):
    #         instances = json_dict["instances"][pos]
    #         for inst in instances:
    #             id_temp.append(inst['video_id'])
    #             label_temp.append(word)
    
    # return pd.DataFrame({'video_id': id_temp,
    #                      'label':label_temp})

    # path = "./datasets/WLASL/start_kit/videos"
    # video_list = get_list_videos(path,"WLASL")

    # video_list['video_id'] = video_list['video_path'].apply(lambda x: x.split('/')[-1].split('.')[0])
    # #video_list['video_id'] = [path.split('/')[-1].split('.')[0] for path in video_list['video_path']]

    # json_dict = pd.read_json("./datasets/WLASL/start_kit/WLASL_v0.3.json")
    
    # json_simply = simplyfy_json_dict(json_dict)

    # video_list = pd.merge(video_list, json_simply,on='video_id')

    # #del video_list["video_id"]
    # return video_list


