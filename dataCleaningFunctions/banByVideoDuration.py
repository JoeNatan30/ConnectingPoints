import pandas as pd
import os
import cv2

general_list = pd.read_csv("results.csv")

video_list = []

unique_glosses = list(set(general_list["classes"]))

for folder, _, files in os.walk(os.path.normpath("../datasets/")):

    files = [_file for _file in files if _file[-3:]=='mp4']

    for file in files:

        comp = os.sep.join([folder.split(os.sep)[-1],file])

        is_in = len(general_list[general_list["videoName"] == comp]) != 0

        if not is_in:
            continue

        origin_path = os.sep.join([folder,file])

        video_list.append((origin_path, comp))

video_df = pd.DataFrame(video_list, columns=['video_path','video_name'])

ban_list = []

for path, name in zip(video_df['video_path'], video_df['video_name']):

    cap = cv2.VideoCapture(path)

    if (cap.isOpened() is False):
        print("Unable to read camera feed", path)

    fps = cap.get(cv2.CAP_PROP_FPS)      # OpenCV v2.x used "CV_CAP_PROP_FPS"
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count/fps


    if duration <= 0.20:
        ban_list.append(name)

pd.DataFrame(ban_list).to_csv("banned_videos_by_duration.csv", header=False, index=False)
