# ConnectingPoints

Repository used to unify keypoints estimators output into a single format, thus making them comparable and analysable.

## Datasets thats works:
 - AEC
 (videos are taken from datasets/AEC/Videos/SEGMENTED_SIGN)
 - PUCP_PSL_DGI156
 (videos are taken from datasets/PUCP_PSL_DGI156/Videos/cropped)
 - WLASL
 (need to download, follow the WLASL guide in [WLALS link](https://github.com/dxli94/WLASL))

 More information about AEC and PUCP_PSL_DGI156 in [PeruSIL](https://github.com/gissemari/PeruvianSignLanguage)
------
## Keypoint estimators available
 - Mediapipe
 - Openpose
 - Wholepose

------
 ## Output structure

There will be one output for each keypoint estimator used and data set (and mixes) chosen
it is need you to create "output" folder

the output is an H5 file (.hdf5) with this structure:

group("#") where # is an index

each group will have:
- ["video_name"] = name of the video
- ["label"] = isolated signs showed in the video
- ["data"] = an TxCxK structure 

where:
- T = timestep of the video
- C = x and y coords
- K = keypoints (with this order: "pose", "face", "left hand" and "right hand")
***
# Dataset download and Models installation

## prepare the enviroment

create a new conda enviroment with python 3.8 (recommended)
then run:
```
sh requirements.sh
```
## download datasets

run:
```
sh create_folders_and_dataset.sh
```
remember that WLASL have additional steps

## model installation
 - **Mediapipe**

This library is already installed if you did the "prepare the enviroment" section.
It also be installing by running:
 ```
 pip install mediapipe
 ```
 - **Wholepose**

 go to keypointEstimators/models/wholepose
 Run:
 ```
 sh download_model.sh
 ```
 this will download "wholebody_hrnet_w48_384x384" pretrained model of [Coco-Wholebody github](https://github.com/jin-s13/COCO-WholeBody)

 - **OpenPose**

 Go to keypointEstimators/models/

 Run:
 ```
 sh openpose_installation.sh
 ```
 This will download the [openpose repository] and then install it with the requiered dependencies to use openpose in python.

 If you have another Nvidia architecture that not correspont with the one in cmake, please modify **cmake parameters**

 then add this line at the beginning of the ".bashrc" (linux)
 ```
 export CUDA_PATH=/usr/local/cuda
 export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH
 export PYTHONPATH=/usr/local/python/openpose:$PYTHONPATH
 ```

 if you are using miniconda, also add
 ```
 export PATH="/home/<user>/miniconda3/bin:$PATH"
 ```
 in case of anaconda, replace "miniconda" with it (check if the version correspond with the number used)

***
# How to use it

Just run:
```
python preprocess.py
```
Select the dataset you want to mix (or to process isolated)
And then, select the keypoint estimator model (only one per run)

The output will be in "output" folder.