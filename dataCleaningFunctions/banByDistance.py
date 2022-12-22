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

data_loss = {
    "gloss":[],
    "total":[],
    "total-val":[],
    "total-train":[],
    "cleaned":[],
    "cleaned-val":[],
    "cleaned-train":[]
}

for unique_class in unique_classes:

    print(unique_class)
    data_loss["gloss"].append(unique_class)
    class_data = data[data["classes"]==unique_class]

    distance = class_data.distance
    
    for_train = class_data[class_data["splitType"]=='train']
    for_val = class_data[class_data["splitType"]=='val']
    
    data_loss["total"].append(len(distance))
    data_loss["total-train"].append(len(for_train))
    data_loss["total-val"].append(len(for_val))

    quartiles = s.quantiles(distance, n=10)
    pivot = quartiles[8]

    print("Pivot", pivot)
    to_ban= class_data[class_data["distance"] >= pivot]
    print(len(to_ban))

    for_train_banned = to_ban[to_ban["splitType"]=='train']
    for_val_banned = to_ban[to_ban["splitType"]=='val']
    data_loss["cleaned-train"].append(len(for_train)-len(for_train_banned))
    data_loss["cleaned-val"].append(len(for_val)-len(for_val_banned))
    
    rest = len(distance) - len(to_ban)
    data_loss["cleaned"].append(rest)

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
ban_list.to_csv("banned_videos_by_distance.csv", header=False, index=False)

pd.DataFrame(data_loss).to_csv("counts.csv",index=False)