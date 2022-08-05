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

def get_keypoint_from_estimator(path, label, kpoint_est_opt):
    st = time.time()
    results = {"openpose": [],
               "mediapipe": [],
               "wholepose": [],
               "label":[]}
    
    functions = {"openpose": openpose_functions,
                 "mediapipe": mediapipe_functions,
                 "wholepose": wholepose_functions}

    cap = cv2.VideoCapture(path)

    if (cap.isOpened() is False):
        print("Unable to read camera feed", path)
        for dataset in kpoint_est_opt:
                results[dataset].append([])
        return results

    print("processing", path)
    ret, frame = cap.read()

    while ret is True:
    
        for dataset in kpoint_est_opt:
            videos_df = get_keypoint_estimation_data(functions[dataset], frame)
            results[dataset].append(videos_df)

        ret, frame = cap.read()
    
    results = [v for k, v in results.items() if v != []]
    results.append(label)
    results.append(path.split(os.sep)[-1])
    et = time.time()
    elapsed_time = et - st
    print('Execution time:', elapsed_time, 'seconds')
    return results

def partial_save(output, partial_output_name, kpoint_est_opt):

    for estimator in kpoint_est_opt:

        name = '--'.join([partial_output_name, estimator])
        to_save = pd.DataFrame({'data':list(output[estimator]),
                                'label':list(output['label']),
                                'video_name':list(output['video_name'])})

        to_save.to_pickle('./output/'+name+'.pk')
        #rn = pd.read_json('./output/'+name+'.json')
        #print(rn)

def get_keypoint_estimator_standarized_output(kpoint_est_opt, data, partial_output_name):
 
    my_array = np.arange(len(data))
    n = 100

    workers = 1 #os.cpu_count()

    splited_data = np.array_split(data, range(n,len(my_array), n))

    acum = []

    for split in splited_data:

        p = Pool(workers)
        kp_list = p.starmap(get_keypoint_from_estimator, zip(split['video_path'],split['label'], repeat(kpoint_est_opt)))
        kp_list = pd.DataFrame(kp_list, columns=kpoint_est_opt + ["label"] + ["video_name"])
        print(kp_list)
        acum.append(kp_list)

        output = pd.concat(acum, ignore_index=True)
        partial_save(output, partial_output_name, kpoint_est_opt)

    return splited_data #pd.concat(dataset_list, ignore_index=True)


dataset_opt = cs.select_datasets()
#dataset_opt = ["AEC"]#["WLASL", "AEC", "PUCP_PSL_DGI156"]
data = get_datasets_data(dataset_opt)

kpoint_est_opt = cs.select_keypoint_estimator()
nani = get_keypoint_estimator_standarized_output(kpoint_est_opt, data, '-'.join(dataset_opt))

