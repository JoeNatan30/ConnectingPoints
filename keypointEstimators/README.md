# Keypoint estimators

This folder have the function used in preprocess.py (in root folder) to allow to use each keypoint estimation:

* Openpose
* Mediapipe
* Wholepose

These models or libraries need to be download or installed in your enviroment to use it.

the function that are common in each file are the following:

- def model_init(...)
- def format_model_output(...)
- def close_model(...)
- def frame_process(...)

-----------------
# Openpose

Install this library directly by [openpose github](https://github.com/CMU-Perceptual-Computing-Lab/openpose.git) or use "openpose_installation.sh" in ("keypointEstimators/models/" folder).

Note: if you use the "openpose_installation.sh" file, you probably need to modify some parameters in line 19 (cmake ...) in order to adapt it to your enviroment.

Also remenber to modify "model_folder" param in openpose_function.py" if you installed openpose in other folder than the default in this file.
-----------------
# Mediapipe

install this library by using ´´´pip install mediapipe´´´ or check it in mediapipe website to get an specific version you want to use.
-----------------
# Wholepose

go to "keypointEstimators/models/wholepose/" folder and download there this file:

- hrnet_w48_coco_wholebody_384x288-6e061c6a_20200922.pth from [link](https://drive.google.com/file/d/1f_c3uKTDQ4DR3CrwMSI8qdsTKJvKVt7p/view)

Acknowledge: We use the code of [this module](https://github.com/jackyjsy/data-prepare/tree/89b556b0cb49a5a401ed939e3977c101df912257/wholepose) in order to adapt it to our work.