import os
import numpy as np

try:
    import pyopenpose as op
except:
    print("comment this 'keypointEstimators/openpose_function.py' in case the code stops")


def model_init():

    opWrapper = op.WrapperPython()

    params = dict()
    params["hand"] = True               # Enable hand keypoint detection
    params["face"] = True               # Enable face keypoint detection
    params['num_gpu'] = 1
    params['num_gpu_start'] = 0
    params['number_people_max'] = 1
    #params["face_detector"] = 0         # Openpose Face rectangle detector
    params["keypoint_scale"] = 3
    params["body"] = 1                  # Enable/Disable Body Keypoint detection
    params["model_folder"] = dirname = os.path.dirname(__file__) + "/models/openpose/models"
    params['render_pose'] = 0
    params["display"] = 0

    opWrapper.configure(params)
    #opWrapper.configure(params=op.get_params_from_file("../../../config/openpose.ini"))
    opWrapper.start()

    return opWrapper

def format_model_output(output):
    newFormat = []
  
    pose = output['pose']
    face = output['face']
    left_hand = output['left_hand']
    right_hand = output['right_hand']

    newFormat.append(pose)
    newFormat.append(face)
    newFormat.append(left_hand)
    newFormat.append(right_hand)

    x = np.asarray([item[0] for sublist in newFormat for item in sublist])
    y = np.asarray([item[1] for sublist in newFormat for item in sublist])

    out = np.asarray([x,y])
    return out

def close_model(opWrapper):
    opWrapper.stop()

def frame_process(opWrapper, frame):

    datum = op.Datum()
    datum.cvInputData = frame
    opWrapper.emplaceAndPop(op.VectorDatum([datum]))

    kpDict = {}
    if not datum.poseKeypoints is None:
        kpDict["pose"] = [[coord[0], coord[1]] for points in datum.poseKeypoints for coord in points]
    else:
        kpDict["pose"] = [ [0.0, 0.0] for point in range(0, 25)]

    if not datum.faceKeypoints is None:
        kpDict["face"] = [[coord[0], coord[1]] for points in datum.faceKeypoints for coord in points]
    else:
        kpDict["face"] = [ [0.0, 0.0] for point in range(0, 70)]

    if not datum.handKeypoints[0] is None:
        kpDict["left_hand"] = [[coord[0], coord[1]] for points in datum.handKeypoints[0] for coord in points]
    else:
        kpDict["left_hand"] = [ [0.0, 0.0] for point in range(0, 21)]

    if not datum.handKeypoints[1] is None:
        kpDict["right_hand"] = [[coord[0], coord[1]] for points in datum.handKeypoints[1] for coord in points]
    else:
        kpDict["right_hand"] = [ [0.0, 0.0] for point in range(0, 21)]

    return kpDict
