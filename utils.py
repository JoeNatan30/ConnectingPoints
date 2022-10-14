import h5py


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