import os
import pyopenpose as op


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

def format_model_output(model_output):
    #print("format!")
    return model_output

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
