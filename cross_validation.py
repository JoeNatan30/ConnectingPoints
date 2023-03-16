from utils import create_df
import pandas as pd
import numpy as np
import h5py
from sklearn.model_selection import StratifiedKFold

def save_data(df, dataset_type, indexs, i):

    out_path = f'./crossValidation/AEC-DGI156-DGI305--mediapipe-{dataset_type}-{i}.hdf5'
    h5_file = h5py.File(out_path, 'w')

    for pos, data_index in enumerate(indexs):
        grupo_name = f"{pos}"
        h5_file.create_group(grupo_name)

        row = df.loc[data_index]

        h5_file[grupo_name]['video_name'] = row.videoName # video name (str)
        h5_file[grupo_name]['label'] = row.classes # classes (str)
        h5_file[grupo_name]['data'] = row.data # data (Matrix)
    h5_file.close()


df_train = create_df("split/AEC-DGI156-DGI305--mediapipe-Train.hdf5")
df_val = create_df("split/AEC-DGI156-DGI305--mediapipe-Val.hdf5")

df = df_train.append(df_val).reset_index()
del df['index']

classes = list(set(df['classes']))
meaning = {k:v for k, v in enumerate(classes)}
meaning_inv = {v:k for k,v in meaning.items()}

X = np.array(df.index.to_numpy())
y = np.array([meaning_inv[_val] for _val in df['classes']])

skf = StratifiedKFold(n_splits=5, random_state=1, shuffle=True)
skf.get_n_splits(X, y)

for i, (train_index, val_index) in enumerate(skf.split(X, y)):
    print(f"Fold {i}:")
    print(f"  Train: index={train_index}")
    print(f"  Test:  index={val_index}")

    save_data(df, 'Train', train_index, i)
    save_data(df, 'Val', val_index, i)

print(df.loc[0]['data'])