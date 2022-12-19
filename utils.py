import h5py
import pandas as pd

def read_h5(path):

    classes = []
    videoName = []
    data = []

    #read file
    with h5py.File(path, "r") as f:
        for index in f.keys():
            classes.append(f[index]['label'][...].item().decode('utf-8'))
            videoName.append(f[index]['video_name'][...].item().decode('utf-8'))
            data.append(f[index]["data"][...])
    
    return classes, videoName, data


def create_df(path):

    glosses, videoName, data = read_h5(path)

    df = pd.DataFrame.from_dict({
        "classes":glosses,
        "videoName": videoName,
        "data":data,  
    })
    return df