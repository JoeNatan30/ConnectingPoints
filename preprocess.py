# Standard library imports
from multiprocessing import Pool
#from itertools import repeat
import os
import warnings

# Third party imports
import pandas as pd
import numpy as np
import cv2
import time
import h5py

# Local imports
import commandSystem as cs
from datasetVideoReader import AEC_videoReader, PUCP_videoReader, WASL_videoReader, PUCP_PSL_DGI305_videoReader
from datasetVideoReader import AUTSL_videoReader, INCLUDE_VideoReader

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
                    "PUCP_PSL_DGI305": PUCP_PSL_DGI305_videoReader,
                    "INCLUDE": INCLUDE_VideoReader,
                    "AUTSL": AUTSL_videoReader,
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
        return np.array([])

    print("processing", path)
    ret, frame = cap.read()
    
    while ret is True:
    
        videos_df = get_keypoint_estimation_data(keypoint_estimator, model, frame)
        results[kp_est_chosed].append(videos_df)

        ret, frame = cap.read()

    results = [np.array(v) for k, v in results.items() if v != []]
    #results.append(label)
    #results.append(path.split(os.sep)[-1])
    et = time.time()
    elapsed_time = et - st
    print('Execution time:', elapsed_time, 'seconds')
    return np.squeeze(np.array(results), axis=0)

def partial_save(output, partial_output_name, estimator):

    name = '--'.join([partial_output_name, estimator])

    os.makedirs("./output", exist_ok=True)
    output.to_pickle('./output/'+name+'.pk')


def get_keypoint_estimator_standarized_output(kpoint_est_opt, data, partial_output_name):

    dataset_name = partial_output_name

    partial_output_name = '-'.join(partial_output_name)

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
 
    num = 0

    save_path = f"./output/{partial_output_name}--{kp_est_chosed}.hdf5"

    h5_file = h5py.File(save_path, 'w')

    # For each video (path)
    for path, label in zip(data['video_path'], data['label']):

        # Where the video is processed
        kp_list = get_keypoint_from_estimator(path, label, model, kp_est_chosed, keypoint_estimator)

        if len(kp_list) == 0:
            print("Error in:", path)
            continue

        # accumulate data
        grupo_name = f"{num}"
        h5_file.create_group(grupo_name)
        # TODO hacer que se devuelva el valor de todo el path desde la carpeta del dataset
        h5_file[grupo_name]['video_name'] = path.split(dataset_name[0])[-1][1:]
        h5_file[grupo_name]['label'] = label
        h5_file[grupo_name]['data'] = np.asarray(kp_list)

        num = num + 1
        print(f"Video processed: {num}")
        
    h5_file.close()

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
get_keypoint_estimator_standarized_output(kpoint_est_opt, data, dataset_opt)