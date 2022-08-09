import pandas as pd

val = pd.read_pickle("./output/PUCP_PSL_DGI156--mediapipe.pk")
print(len(val.T["data"]))