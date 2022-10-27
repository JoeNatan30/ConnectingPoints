# Standard
import sys

# Three-party
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns
from fastdtw import fastdtw
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean

# Local
sys.path.append('../')
from utils import read_h5

def save_in_h5(df, h5, index, instanceDistances):
    grupo_name = f"{index}"
    h5.create_group(grupo_name)
    h5[grupo_name]['video_name'] = df.iloc[index]['videoName']
    h5[grupo_name]['label'] = df.iloc[index]['classes']
    h5[grupo_name]["splitType"] = df.iloc[index]["splitType"]
    h5[grupo_name]['distances'] = np.array(instanceDistances)

    return h5

def DTW_process(df, current_data, compare_index):

    keypointsDistances = []

    # To go through each timeline of a N° keypoint
    for posPoint in range(current_data.shape[0]):

        compare_data = df.iloc[compare_index]['data']

        pointDist, _ = fastdtw(current_data[posPoint],compare_data[posPoint] , dist=euclidean)
        keypointsDistances.append(pointDist)

    return np.mean(keypointsDistances)

classes, videoName, data = read_h5('../split/AEC-DGI156-DGI305--mediapipe-Train.hdf5')
classes_val, videoName_val, data_val = read_h5('../split/AEC-DGI156-DGI305--mediapipe-Val.hdf5')

# To know the type of split it have
splitType = ["train" for _ in range(len(videoName))]
splitType = splitType + ["val" for _ in range(len(videoName_val))]

classes = classes + classes_val
videoName = videoName + videoName_val
data = data + data_val


del(data_val)
del(videoName_val)
del(classes_val)

# To select only desired points
points = pd.read_csv(f"./points_71.csv")
points = np.array(points.mp_pos)-1
points = list(points)

# to have a list of unique classes
unique_classes = list(set(classes))

# Process to have the data in a dataframe
df = pd.DataFrame.from_dict({
    "classes":classes,
    "videoName": videoName,
    "splitType": splitType,
            # shape = N°keypoints x N°TimeSteps x Coords
    "data": [np.moveaxis(d[:,:,points], -1, 0) for d in data],  
})

distByInstance = {
    'classes':[],
    'distance':[],
    'videoName':[],
    'splitType':[]
}

generalDensity = []

unique_classes = [_unique for _unique in unique_classes if len(df[df["classes"]==_unique].index) >= 15]

#print(unique_classes)
print(len(unique_classes))

h5_file = h5py.File("./distances-detail.hdf5", 'w')

# To get a group of instance of a unique class
for unique_class in tqdm(unique_classes):

    #print('#'*50)
    #print(unique_class)

    unique_df = df[df["classes"]==unique_class].index

    num_instance = len(unique_df)

    # If the class only have one instance
    if num_instance <= 1:
        distanceMean = 0.0
        continue

    classDistance = []

    # For each instance of the group (use index to retrieve the specific data)
    for current_index in tqdm(unique_df):
        current_data = df.iloc[current_index]['data']

        instanceDistances = []

        # To go around each instance of the group
        for compare_index in unique_df:

            # To avoid process the same instance in the comparison
            if compare_index == current_index:
                continue

            keypointsDistance = DTW_process(df, current_data, compare_index)

            # Mean of all keypoints time series to get a general instance distance
            instanceDistances.append(keypointsDistance)

        h5_file = save_in_h5(df, h5_file, current_index, instanceDistances)

        instanceDistMean = np.mean(instanceDistances)

        distByInstance['classes'].append(df.iloc[current_index]['classes'])
        distByInstance['distance'].append(instanceDistMean)
        distByInstance['videoName'].append(df.iloc[current_index]['videoName'])
        distByInstance['splitType'].append(df.iloc[current_index]['splitType'])
        
        # add the instance distance mean to have a general Class distance
        classDistance.append(instanceDistMean)

        #print(df.iloc[current_index]['classes'], distanceMean)
    
    classDistMean = np.mean(classDistance)        

    generalDensity.append(classDistMean)
    pd.DataFrame.from_dict(distByInstance, orient='index').T.to_csv("./results.csv")

# set a grey background (use sns.set_theme() if seaborn version 0.11.0 or above) 
sns.set(style="darkgrid")

sns.distplot(generalDensity)
# in the next version of the distplot function, one would have to write:
# sns.distplot(data=df, x="sepal_length", kind='hist') # note that 'kind' is 'hist' by default
plt.show()

pd.DataFrame.from_dict(distByInstance, orient='index').T.to_csv("./results.csv")
h5_file.close()