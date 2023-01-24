import h5py
from collections import Counter
from sklearn.model_selection  import train_test_split

class DataReader():

    def __init__(self, path):

        self.classes = []
        self.videoName = []
        self.data = []
        self.path = path

        #read file
        with h5py.File(path, "r") as f:
            for index in f.keys():
                self.classes.append(f[index]['label'][...].item().decode('utf-8'))
                self.videoName.append(f[index]['video_name'][...].item().decode('utf-8'))
                self.data.append(f[index]["data"][...])
    
    def generate_meaning_dict(self):

        meaning = {v:k for (k,v) in enumerate(set(self.classes))}
        self.labels = [meaning[_class] for _class in self.classes]

    def selectInstances(self, selected):
    
        classes = []
        videoName = []
        data = []

        for pos in range(len(self.classes)):
            #To avoid no selected classes
            if self.classes[pos] not in selected:
                continue

            classes.append(self.classes[pos])
            videoName.append(self.videoName[pos])
            data.append(self.data[pos])

        self.classes = classes
        self.videoName = videoName
        self.data = data

    def saveData(self, indexOrder, train=True):

        #reorder data
        class_tmp = [self.classes[pos] for pos in indexOrder]
        videoName_tmp = [self.videoName[pos] for pos in indexOrder]
        data_tmp = [self.data[pos] for pos in indexOrder]
        labels_tmp = [self.labels[pos] for pos in indexOrder]

        # set the path
        save_path = f"split/{self.path.split('/')[1]}"
        save_path = save_path.split('.')

        if train:
            print("Train:", len(indexOrder))
            path = f"{save_path[0]}-Train.hdf5"
        else:
            print("Val:", len(indexOrder))
            path = f"{save_path[0]}-Val.hdf5"

        # Save H5 
        h5_file = h5py.File(path, 'w')

        for pos, (c, v, d, l) in enumerate(zip(class_tmp, videoName_tmp, data_tmp, labels_tmp)):
            grupo_name = f"{pos}"
            h5_file.create_group(grupo_name)
            h5_file[grupo_name]['video_name'] = v # video name (str)
            h5_file[grupo_name]['label'] = c # classes (str)
            h5_file[grupo_name]['data'] = d # data (Matrix)
            #h5_file[grupo_name]['class_number'] = l #label (int)
            
        h5_file.close()


    def splitDataset(self):
        
        # To know the number of instance per clases
        counter = Counter(self.classes)

        # Select the words that have more or equal than 15 instances    
        counter = [word for word, count in counter.items() if count >= 15]
        print(counter)
        # Errase banned words
        bannedList = ["ya", "qué?", "qué", "bien", "dos", "ahí", "luego", "yo", "él", "tú", "???", "NNN", "sí"]
        selected = list(set(counter) - set(bannedList))

        # Filter the data to have selected instances
        self.selectInstances(selected)

        # generate classes number to use it in stratified option
        self.generate_meaning_dict()

        # split the data into Train and Val (but use list position as X to reorder)
        x_pos = range(len(self.labels))
        pos_train, pos_val, y_train, y_val = train_test_split(x_pos, self.labels, train_size=0.8 , random_state=32, stratify=self.labels)
        
        # save the data
        self.saveData(pos_train,train=True)
        self.saveData(pos_val, train=False)

    

for dataset in ["AEC", "WLASL", "PUCP_PSL_DGI156", "PUCP_PSL_DGI305"]:
    for kpModel in ["mediapipe"]: # ["mediapipe", "openpose", "wholepose"]:
        
        print(f"procesing {dataset}-{kpModel}...")
        path = f"output/{dataset}--{kpModel}.hdf5"
        dataReader = DataReader(path)
        dataReader.splitDataset()
        #splitDataset(path)

