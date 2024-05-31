import sys
import csv
from collections import Counter

import pandas as pd

sys.path.append('../')
from utils import read_h5

dataset_name = "PUCP_PSL_DGI305"

glosses, videoName, data = read_h5(f"../output/{dataset_name}--mediapipe.hdf5")

counts = Counter(glosses)

with open(f'total_instances_{dataset_name}.csv', 'w', newline='') as csvfile:
    fieldnames = ["Gloss", "numb"]
    writer = csv.writer(csvfile, delimiter=';') 
    writer.writerow(fieldnames)
    for key, value in counts.items():
        writer.writerow([key, value])