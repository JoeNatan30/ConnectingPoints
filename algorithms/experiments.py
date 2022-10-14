import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

import statistics as s

data = pd.read_csv("results.csv")
path = "histograms"

unique_classes = list(set(data.classes))

after_delete = []
before_delete = []
class_name = []

delete_count = 0
total_count = 0

videos_to_ban = []

for unique_class in unique_classes:
    print(unique_class)

    class_data = data[data["classes"]==unique_class]

    distance = class_data.distance

    quartiles = s.quantiles(distance, n=5)
    qq = (quartiles[1]-quartiles[0])*1.5
    
    iqr = quartiles[1]
    pivot = 1000.0

    if quartiles[1] + qq >= quartiles[2]:

        iqr = quartiles[2]

        if quartiles[2] + qq >= quartiles[3]:
            pivot = quartiles[3] + qq 
        else:
            pivot = quartiles[3]
    else:
        pivot = quartiles[2]

    print("Pivot", pivot)
    to_ban= class_data[class_data["distance"] >= pivot]
    print(len(to_ban))

    rest = len(distance) - len(to_ban)

    after_delete.append(rest)
    before_delete.append(len(distance))
    class_name.append(unique_class)

    delete_count = delete_count + len(to_ban)
    total_count = total_count + len(distance)

    videos_to_ban = videos_to_ban + list(to_ban["videoName"])

byClassVariance = data.groupby('classes')['distance'].var()


print({gloss: variance for gloss, variance in byClassVariance.items() if variance > 8.0})


pd.DataFrame.from_dict({
    "gloss":class_name,
    "after_delete":after_delete,
    "Initial_count":before_delete,
}).to_csv("countAfterQuartilAnalysis.csv",index=False)
print("N° data to delete: ",delete_count,"/",total_count, (delete_count*100)/total_count,"%")


ban_list = pd.DataFrame(videos_to_ban)
print(ban_list)
ban_list.to_csv("video_to_ban.csv", header=False, index=False)