from datasetVideoReader import  AEC_videoReader, PUCP_videoReader, WASL_videoReader
import cv2
import pandas as pd
import numpy as np
from collections import Iterable
import h5py

def flatten(lis):
    for item in lis:
        if isinstance(item, Iterable) and not isinstance(item, str):
            for x in flatten(item):
                yield x
        else:        
            yield item

def check(dataset, kpModel):
    print("Checking... ", dataset, kpModel)
    video_reader = {"AEC": AEC_videoReader,
                    "PUCP_PSL_DGI156": PUCP_videoReader,
                    "WLASL": WASL_videoReader}

    dataPath = video_reader[dataset].get_data()

    classes_h5 = []
    videoName_h5 = []
    data_h5 = []
    index_h5 = []

    with h5py.File(f"./output/{dataset}--{kpModel}.hdf5", "r") as f:
        for index in f.keys():
            classes_h5.append(f[index]['label'][...].item().decode('utf-8'))
            videoName_h5.append(f[index]['video_name'][...].item().decode('utf-8'))
            data_h5.append(np.asarray(f[index]["data"][...]))
            index_h5.append(index)
        


    for path in dataPath['video_path']:
        cap = cv2.VideoCapture(path)

        if (cap.isOpened() is False):
            print("Unable to read camera feed", path)

        ret, frame = cap.read()

        frame_height, frame_width = frame.shape[:2]
        
        maxSize = min(frame_width, frame_height)

        videoName = path.split('/')[-1].strip('\n')

        print("#################################")

        for pos, row in enumerate(videoName_h5):

            if row == videoName:
                print("Found:", row)
                index = pos

        index = int(index)

        print(frame_height, frame_width)
        print(classes_h5[index])
        print(videoName_h5[index])
        

        data = data_h5[index]
        data_flat = flatten(data)
        data_flat = np.array(list(data_flat))
        print(len([_ for _  in data_flat if _ > 1.0]))

        check = np.where(data_flat <= 1.3, True, False)
        check = check.all()
        assert check, "Mayor a 1.1" 

        check = np.where(data_flat >= -0.3, True, False)
        check = check.all()
        assert check, "Menor que -0.1"

dataset = ["PUCP_PSL_DGI156", "AEC", "WLASL"]
kpModel = ["wholepose"]# ["wholepose","mediapipe","openpose"]

for _kpModel in kpModel:
    for _dataset in dataset:
    
        check(_dataset, _kpModel)