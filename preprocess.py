# Standard library imports
from multiprocessing import Pool
from itertools import repeat
import os
import warnings

# Third party imports
import pandas as pd
import numpy as np
import cv2
import time

# Local imports
import commandSystem as cs
from datasetVideoReader import AEC_videoReader, PUCP_videoReader, WASL_videoReader

try:
    from keypointEstimators import mediapipe_functions
except:
    print("Mediapipe is not installed")
    print("Recommendation: Use 'pip install mediapipe'")

try:
    from keypointEstimators import openpose_functions
except:
    print("Openpose is not installed")
    print("Recommendation: use the 'openpose_installation.sh' file in './keypointEstimators/models' to install openpose")

try:
    from keypointEstimators import wholepose_functions
except:
    print("Wholepose is not installed")
    print("Recommendation: use the 'download_model.sh' file in './keypointEstimators/models/wholepose' to download wholepose model")

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

def get_keypoint_estimation_data(keypoint_estimator, model, frame):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        warnings.warn("deprecated", DeprecationWarning)
        #model = keypoint_estimator.model_init()
        frame_kp = keypoint_estimator.frame_process(model, frame)

    return frame_kp

def get_keypoint_from_estimator(path, label, model, kp_est_chosed, keypoint_estimator):
    st = time.time()
    results = {"openpose": [],
               "mediapipe": [],
               "wholepose":[]}

    cap = cv2.VideoCapture(path)

    if (cap.isOpened() is False):
        print("Unable to read camera feed", path)
        results[kp_est_chosed].append([])
        return results, True

    print("processing", path)
    ret, frame = cap.read()
    
    while ret is True:
    
        videos_df = get_keypoint_estimation_data(keypoint_estimator, model, frame)
        results[kp_est_chosed].append(videos_df)

        ret, frame = cap.read()

    results = [v for k, v in results.items() if v != []]
    results.append(label)
    results.append(path.split(os.sep)[-1])
    et = time.time()
    elapsed_time = et - st
    print('Execution time:', elapsed_time, 'seconds')
    return results

def partial_save(output, partial_output_name, estimator):

    name = '--'.join([partial_output_name, estimator])

    os.makedirs("./output", exist_ok=True)
    output.to_pickle('./output/'+name+'.pk')


def get_keypoint_estimator_standarized_output(kpoint_est_opt, data, partial_output_name):

    # Select keypoint stimator
    kp_est_chosed = kpoint_est_opt[0]
    if "openpose" == kp_est_chosed:
        keypoint_estimator = openpose_functions
    elif "mediapipe" == kp_est_chosed:
        keypoint_estimator = mediapipe_functions
    elif "wholepose" == kp_est_chosed:
        keypoint_estimator = wholepose_functions 
    
    # initilize the model
    model = keypoint_estimator.model_init()
 
    # data acumulators
    kp_acum = []
    path_acum = []
    name_acum = []
    label_acum = []

    num = 0

    # For each video (path)
    for path, label in zip(data['video_path'], data['label']):

        # Where the video is processed
        kp_list = get_keypoint_from_estimator(path, label, model, kp_est_chosed, keypoint_estimator)
        
        num = num + 1
        print(f"Video processed: {num}")
        
        # accumulate data
        kp_acum.append(kp_list)
        label_acum.append(label)
        name_acum.append(path.split(os.sep)[-1])

        # Format of the output
        output = pd.DataFrame.from_dict({
            "data":  kp_acum,
            "label": label_acum,
            "name":  name_acum,
        }, orient='index')

        partial_save(output, partial_output_name, kp_est_chosed)

    # close the model (some models no need to close)
    keypoint_estimator.close_model(model)

#this list have this structure
#dataset_opt = ["WLASL", "AEC", "PUCP_PSL_DGI156"]
dataset_opt = cs.select_datasets()

# To obtain a Dataframe with videos paths, labels and video names
data = get_datasets_data(dataset_opt)

# the structure is similar than dataset_opt but it is for keypoint stimator (and just one)
kpoint_est_opt = cs.select_keypoint_estimator()

# To process the data
get_keypoint_estimator_standarized_output(kpoint_est_opt, data, '-'.join(dataset_opt))