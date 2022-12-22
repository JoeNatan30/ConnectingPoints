import sys

import pandas as pd
import matplotlib.pyplot as plt

sys.path.append('../')
from utils import create_df

def get_length(row):
    return len(row)


df_train = create_df("../split/AEC-DGI156-DGI305--mediapipe-Train.hdf5")
df_val = create_df("../split/AEC-DGI156-DGI305--mediapipe-Val.hdf5")

df_train['length'] = df_train['data'].apply(get_length)
df_val['length'] = df_val['data'].apply(get_length)

df_train.boxplot(column=['length'], 
                 by="classes",
                 #title="Training- Distribution by class",
                 rot=90,
                 figsize=(15, 8)).get_figure().savefig('3-dataset-length-training.png')


df_val.boxplot(column=['length'], 
                 by="classes",
                 #title="Training- Distribution by class",
                 rot=90,
                 figsize=(15, 8)).get_figure().savefig('3-dataset-length-validation.png')

print("Training:  ",df_train['length'].mean())
print("Validation:",df_val['length'].mean())
