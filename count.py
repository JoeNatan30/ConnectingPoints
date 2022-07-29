# Standard library imports
from multiprocessing import Pool
from itertools import repeat
import os
import warnings

# Third party imports
import pandas as pd
import numpy as np
import cv2

# Local imports
import commandSystem as cs
from datasetVideoReader import AEC_videoReader, PUCP_videoReader, WASL_videoReader
from keypointEstimators import mediapipe_functions, openpose_functions, wholepose_functions

def get_videoReader_data(videoReader):
    return videoReader.get_data()

def get_datasets_data(dataset_opt):

    video_reader = {"AEC": AEC_videoReader,
                    "PUCP_PSL_DGI156": PUCP_videoReader,
                    "WLASL": WASL_videoReader}

    dataset_list = []

    for dataset in dataset_opt:
        videos_df = get_videoReader_data(video_reader[dataset])
        dataset_list.append(videos_df)

    return pd.concat(dataset_list, ignore_index=True)

def get_keypoint_estimation_data(keypoint_estimator, frame):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        warnings.warn("deprecated", DeprecationWarning)
        model = keypoint_estimator.model_init()
        frame_kp = keypoint_estimator.frame_process(model, frame)
    return frame_kp

def partial_save(output, partial_output_name, kpoint_est_opt):

    for estimator in kpoint_est_opt:

        name = '--'.join([partial_output_name, estimator])
        to_save = pd.DataFrame({'data':list(output[estimator]),
                                'label':list(output['label'])})

        to_save.to_json('./output/'+name+'.json')
        #rn = pd.read_json('./output/'+name+'.json')
        #print(rn)

def check_videos(kpoint_est_opt, data, partial_output_name):
 
    ok = []
    error = []

    for path in data['video_path']:

        cap = cv2.VideoCapture(path)
        if (cap.isOpened() is False):
            error.append(path)
        else:
            ok.append(path)

    return ok, error

dataset_opt = cs.select_datasets()
#dataset_opt = ["AEC"]#["WLASL", "AEC", "PUCP_PSL_DGI156"]
data = get_datasets_data(dataset_opt)

kpoint_est_opt = cs.select_keypoint_estimator()
ok, error = check_videos(kpoint_est_opt, data, '-'.join(dataset_opt))

n_ok = len(ok)
n_err = len(error)

print("IN:", dataset_opt)
print("ok:", n_ok)
print("Error:", n_err)
print(f"\n{(1 - n_err)/(n_ok + n_err)} are correct")