import sys

import pandas as pd

sys.path.append('../')
from utils import read_h5


glosses, videoName, data = read_h5("../split/AEC-DGI156-DGI305--mediapipe-Val.hdf5")

df_val = pd.DataFrame.from_dict({
    "classes":glosses,
    "videoName": videoName,
    "data":data,  
})

glosses, videoName, data = read_h5("../split/AEC-DGI156-DGI305--mediapipe-Train.hdf5")

df_train = pd.DataFrame.from_dict({
    "classes":glosses,
    "videoName": videoName,
    "data":data,  
})

glosses, videoName, data = read_h5("../split/AEC-DGI156-DGI305--mediapipe-Val-comp.hdf5")

df_val_comp = pd.DataFrame.from_dict({
    "classes":glosses,
    "videoName": videoName,
    "data":data,  
})

glosses, videoName, data = read_h5("../split/AEC-DGI156-DGI305--mediapipe-Train-comp.hdf5")

df_train_comp = pd.DataFrame.from_dict({
    "classes":glosses,
    "videoName": videoName,
    "data":data,  
})

print("quitar",set(df_train_comp.classes)-set(df_train.classes))
print("agregar",set(df_train.classes)-set(df_train_comp.classes))
print()