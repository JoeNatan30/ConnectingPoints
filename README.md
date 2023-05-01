# ConnectingPoints

Repository used to unify keypoints estimators output into a single format, thus making them comparable and analysable.

## Datasets thats works:

| Datasets                | Description                                                                                | Download Link                                                                                                   |
|-------------------------|--------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------|
| AEC                     | "Aprendo En Casa" Peruvian Sign Language (LSP), Videos are taken from datasets/AEC/Videos/SEGMENTED_SIGN            | [AEC_link](https://drive.google.com/uc?id=1WHxKijB8t5JLljM59hAqi5KY0U6d7OzA)                                 |
| PUCP_PSL_DGI156         | "PUCP University Peruvian Sign Language Dirección de investigación N° 156", Videos are taken from datasets/PUCP_PSL_DGI156/Videos/cropped                              | [PUCP_PSL_DGI156_link](https://drive.google.com/uc?id=1WHxKijB8t5JLljM59hAqi5KY0U6d7OzA)                          |
| WLASL (Word Level ASL)  | "Word-level American Sign Language". Need to download and follow the WLASL guide                                             | [WLALS link](https://github.com/dxli94/WLASL)                                                                   |
| AUTSL (Turkish SL)      | "The Arkara University Turkish Sign Language"                                                                         | [AUTSL_link](http://cvml.ankara.edu.tr/datasets/)                                                                    |
| INCLUDE (Indian SL)     | "A Large Scale Dataset for Indian Sign Language Recognition"                                                                         | [INCLUDE link](https://zenodo.org/record/4010759)                                                              |


 More information about AEC and PUCP_PSL_DGI156 in [PeruSIL](https://github.com/gissemari/PeruvianSignLanguage)
 
## download AEC and PUCP_DGI_156 dataset

run:
```
sh create_folders_and_dataset.sh
```
The other datasets have to be located manually in the "dataset" folder

------
## Keypoint estimators available
 - **Mediapipe**
 - **Openpose**
 - **Wholepose**

Note: more info about this model installation are in **Keypoint estimator models installation**
------
 ## Output structure

There will be one output for each keypoint estimator used and data set (and mixes) chosen
it is need you to create "output" folder

the output is an H5 file (.hdf5) with this structure:

group("#") where # is an index

each group will have:
- ["video_name"] = name of the video (this includes the relative path that starts after the name of the dataset)
- ["label"] = isolated signs showed in the video
- ["data"] = an TxCxK structure 

video_name example (in AEC dataset): "Videos/SEGMENTED_SIGN/ira_alegria/abuelo_118.mp4"

where:
- T = timestep of the video
- C = x and y coords
- K = keypoints (with this order: "pose", "face", "left hand" and "right hand")
***
# Prepare the environment
create a new conda enviroment with python 3.8 (recommended)
then run:
```
sh requirements.sh
```

# Keypoint estimator models installation

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
 this will download "wholebody_hrnet_w48_384x384" pretrained model of [Coco-Wholebody github](https://github.com/jin-s13/COCO-WholeBody). The link was obtained from (Smile-lab)[https://github.com/jackyjsy/CVPR21Chal-SLR]

 - **OpenPose**

 Go to keypointEstimators/models/

 Run:
 ```
 sh openpose_installation.sh
 ```
 This will download the [openpose repository](https://github.com/CMU-Perceptual-Computing-Lab/openpose) and then install it with the requiered dependencies to use openpose in python.

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
