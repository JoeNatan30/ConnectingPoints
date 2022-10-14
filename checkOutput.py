from msilib import datasizemask
import pandas as pd
from utils import read_h5


glosses, videoName, data = read_h5("output/PUCP_PSL_DGI156--mediapipe.hdf5")

dataset = pd.DataFrame({
    "glosses":glosses,
    "videoName":videoName,
    "data":data
})

print(dataset)