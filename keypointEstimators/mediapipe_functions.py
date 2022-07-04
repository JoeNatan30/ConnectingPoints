# Standard library imports
import os 

# Third party imports
import mediapipe as mp
import numpy as np
import cv2

# Local imports

def model_init(static_image_mode = True, ):

    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(
        static_image_mode= static_image_mode,
        model_complexity=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)

    print("init MediaPipe model - Done")
    return holistic

def frame_process(frame):

    imageBGR = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    height, width, _ = imageBGR.shape

    holisResults = holistic.process(imageBGR)

    kpDict = {}

    #POSE
    kpDict["pose"]={}

    if holisResults.pose_landmarks:
        kpDict["pose"] = [ [point.x, point.y] for point in holisResults.pose_landmarks.landmark]
    else:
        kpDict["pose"] = [ [0.0, 0.0] for point in range(0, 33)]

    # HANDS

    # Left hand
    kpDict["left_hand"]={}

    if(holisResults.left_hand_landmarks):
        kpDict["left_hand"] = [ [point.x, point.y] for point in holisResults.left_hand_landmarks.landmark]
    else:
        kpDict["left_hand"] = [ [0.0, 0.0] for point in range(0, 21)]

    # Right hand
    kpDict["right_hand"]={}

    if(holisResults.right_hand_landmarks):
        kpDict["right_hand"] = [ [point.x, point.y] for point in holisResults.right_hand_landmarks.landmark]

    else:
        kpDict["right_hand"] = [ [0.0, 0.0] for point in range(0, 21)]

    # Face mesh
    kpDict["face"]={}

    if(holisResults.face_landmarks):

        kpDict["face"] = [ [point.x, point.y] for point in holisResults.face_landmarks.landmark]

    else:
        kpDict["face"] = [[0.0, 0.0] for point in range(0, 468)]

    print("frame_process")
    return kpDict

'''
holistic = model_init()
cap = cv2.VideoCapture(os.getcwd()+'/datasets/???_18.mp4')
ret, frame = cap.read()

#frame = np.random.rand(100,100,3)
frame_kp = frame_process(frame)
print(frame_kp)
holistic.close()
'''