import sys

import pandas as pd
import matplotlib.pyplot as plt

sys.path.append('../')
from utils import create_df


def create_dataset_info(row):

    dataset_name = 'DGI305'
    if 'proteinas_porcentajes' in row:
        dataset_name = 'AEC'
    if 'ira_alegria' in row:
        dataset_name = 'AEC'
    if 'Historias_vinetas_' in row:
        dataset_name = 'DGI156'

    return dataset_name

def percentage_per_dataset(row):

    total = row['AEC'] + row['DGI156'] + row['DGI305']
    row['AEC'] = row['AEC']/total
    row['DGI156'] = row['DGI156']/total
    row['DGI305'] = row['DGI305']/total

    return row


df_train = create_df("../split/AEC-DGI156-DGI305--mediapipe-Train.hdf5")
df_val = create_df("../split/AEC-DGI156-DGI305--mediapipe-Val.hdf5")

df_train['dataset'] = df_train['videoName'].apply(lambda x: create_dataset_info(x))
df_val['dataset'] = df_val['videoName'].apply(lambda x: create_dataset_info(x))

pd_mix_train = pd.concat([df_train, pd.DataFrame({'split':'train'}, index=df_train.index)], axis=1)
pd_mix_val = pd.concat([df_val, pd.DataFrame({'split':'   val'}, index=df_val.index)], axis=1)
pd_mix = pd_mix_train.append(pd_mix_val)

#fig = class_counts.plot(kind='bar',  figsize=(20, 16), ).get_figure()

#################################
##  BAR - SPLIT DISTRIBUTION

fig = plt.figure()
pd_split_count = pd_mix.groupby(['dataset','split']).data.count().unstack().reset_index()
ax = pd_split_count.plot.bar(
                    x='dataset',
                    title="Split distribution",
                    rot=30,
                    figsize=(13, 8))

for container in ax.containers:
    ax.legend(loc='best')
    ax.bar_label(container)

ax.get_figure().savefig('3-dataset-split-distribution.png')

#################################
##  BAR - class with split DISTRIBUTION

fig, axes = plt.subplots(nrows=2, ncols=1) 
plt.subplots_adjust(wspace=0.5, hspace=0.5)

df_train_groupby = pd_mix_train.groupby(['classes', 'dataset']).dataset.count().unstack().reset_index()
df_train_groupby = df_train_groupby.fillna(0.0)


df_train_groupby.plot(kind="bar",
                      x='classes',
                      ax=axes[0],
                      title="Training - Distribution by class",
                      y=["AEC","DGI156","DGI305"],
                      figsize=(20, 10),
                      stacked=True)

df_val_groupby = pd_mix_val.groupby(['classes', 'dataset']).dataset.count().unstack().reset_index()
df_val_groupby = df_val_groupby.fillna(0.0)



df_val_groupby.plot(kind="bar",
                    x='classes',
                    ax=axes[1],
                    title="Validation - Distribution by class",
                    y=["AEC","DGI156","DGI305"],
                    figsize=(20, 10),
                    stacked=True)
fig.savefig('3-dataset-byClass-split.png')

#################################
##  BAR - class with split DISTRIBUTION  PERCENTAGE

fig, axes = plt.subplots(nrows=2, ncols=1) 
plt.subplots_adjust(wspace=0.5, hspace=0.5)

df_train_groupby_perc = df_train_groupby.apply(percentage_per_dataset ,axis=1)
df_train_groupby_perc.plot(kind="bar",
                      x='classes',
                      ax=axes[0],
                      title="Training - Distribution by class",
                      y=["AEC","DGI156","DGI305"],
                      figsize=(20, 10),
                      stacked=True)

df_val_groupby_perc = df_val_groupby.apply(percentage_per_dataset ,axis=1)
df_val_groupby_perc.plot(kind="bar",
                    x='classes',
                    ax=axes[1],
                    title="Validation - Distribution by class",
                    y=["AEC","DGI156","DGI305"],
                    figsize=(20, 10),
                    stacked=True)

fig.savefig('3-dataset-byClass-split-perc.png')
