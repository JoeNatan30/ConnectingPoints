import pyopenpose as op

def model_init():

    opWrapper = op.WrapperPython()

    params = dict()
    params["hand"] = True               # Enable hand keypoint detection
    params["face"] = True               # Enable face keypoint detection
    params["face_detector"] = 0         # Openpose Face rectangle detector 
    params["body"] = 1                  # Enable/Disable Body Keypoint detection
    params["model_folder"] = "./models/openpose/models"
    params['render_pose'] = 0
    params["display"] = 0

    opWrapper.configure(params)
    #opWrapper.configure(params=op.get_params_from_file("../../../config/openpose.ini"))
    opWrapper.start()

    return opWrapper

def format_model_output(model_output):
    #print("format!")
    return model_output


def frame_process(opWrapper, frame):

    datum = op.Datum()
    datum.cvInputData = frame
    opWrapper.emplaceAndPop([datum])

    data = datum.poseKeypoints
    print()


    opWrapper.stop()
    opWrapper.close()


    return data