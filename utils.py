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

def save_h5_new_format(path, new_path):
    #path = "split/AEC--mediapipe-Train.hdf5"

    df = utils.create_df(path)

    #save_path = "split/new_format/AEC--mediapipe-Train.hdf5"

    h5_file = h5py.File(new_path, 'w')

    for section in df:

        group = h5_file.create_group(section)

        if section == 'classes' or section == 'videoName':
            muestra = [str(_value) for _value in df[section]]
            group.create_dataset('values', data=muestra)
        if section == 'data':
            muestra = [np.array(_value).astype('f8') for _value in df[section]]
            for pos, sample in enumerate(muestra):
                h5_file[section][str(pos)] = sample

        #h5_file[grupo_name] = df[section]

    h5_file.close()

def retrieve_h5_data_new_format(path):
    dataset = {}

    with h5py.File(save_path, "r") as f:
            for section in f.keys():
                if section == 'classes' or section == 'videoName':
                    dataset[section] = np.array([i[:].decode('utf-8') for i in f[section]['values']])
                if section == 'data':
                    dataset[section] = np.array([i[:] for k,i in f[section].items()])

    df = pd.DataFrame.from_dict(dataset)
    return df

