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

# def get_h5_last_position(h5_file):
#     groups = list(h5_file.keys())
#     posiciones = [int(nombre) for nombre in groups]
#     return max(posiciones)

def backup_data(file, backup):
    print("generating dataset backup...")
    groups = list(backup.keys())

    for group_name in groups:
        _data = backup[group_name]

        file.create_group(group_name)
        file[group_name]['video_name'] = _data['video_name'][...].item().decode('utf-8')
        file[group_name]['label'] = _data['label'][...].item()
        file[group_name]['data'] = _data["data"][...]
    
    backup.close()

    return [int(nombre) for nombre in groups] if groups != None else []

def initialize_h5_file(save_path):
    """Inicializa el archivo H5 dependiendo de su existencia y la respuesta del usuario."""

    # If its a new file
    if not os.path.exists(save_path):
        print("Creating a new H5 file...")
        h5_file = h5py.File(save_path, 'w')
    
        return h5_file, 0
    
    else:
        # Ask for continue processing
        #answer = cs.select_continue_processing()

        save_path_old = f"{'.'.join(save_path.split('.')[:-1])}_old.hdf5"
    
        os.replace(save_path, save_path_old)

        h5_file_old = h5py.File(save_path_old, "r")
        h5_file = h5py.File(save_path, "w")

        positions_list = backup_data(h5_file, h5_file_old)

        if len(positions_list) == 0:
            last_position = 0
        else:
            last_position = max(positions_list)

        return h5_file, last_position
    
def save_data_h5(h5_file, num, path, dataset_name, kp_list, label):

    grupo_name = f"{num}"
    h5_file.create_group(grupo_name)
    # TODO hacer que se devuelva el valor de todo el path desde la carpeta del dataset
    h5_file[grupo_name]['video_name'] = path #.split(dataset_name[0])[-1][1:]
    h5_file[grupo_name]['label'] = label
    h5_file[grupo_name]['data'] = np.asarray(kp_list)

def select_keypoint_estimator(kp_est_chosed):
    
    if "openpose" == kp_est_chosed:
        keypoint_estimator = openpose_functions
    elif "mediapipe" == kp_est_chosed:
        keypoint_estimator = mediapipe_functions
    elif "wholepose" == kp_est_chosed:
        keypoint_estimator = wholepose_functions
    
    return keypoint_estimator

def get_keypoint_estimator_standarized_output(kpoint_est_opt, data, partial_output_name):

    dataset_name = partial_output_name

    partial_output_name = '-'.join(partial_output_name)

    # Select keypoint stimator
    kp_est_chosed = kpoint_est_opt[0]
    keypoint_estimator = select_keypoint_estimator(kp_est_chosed)

    save_path = f"./output/{partial_output_name}--{kp_est_chosed}.hdf5"

    # if the H5 file exists
    h5_file, last_position = initialize_h5_file(save_path)

    # initilize the model
    model = keypoint_estimator.model_init()
 
    num = 0

    # For each video (path)
    for path, label in zip(data['video_path'], data['label']):

        if num <= last_position and last_position != 0:
            print(f"already processed video: {num}")
            num = num + 1
            continue

        # Where the video is processed
        kp_list = get_keypoint_from_estimator(path, label, model, kp_est_chosed, keypoint_estimator)

        if len(kp_list) == 0:
            print("Error in:", path)
            continue

        # accumulate data
        save_data_h5(h5_file, num, path, dataset_name, kp_list, label)

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