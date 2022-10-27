import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

import statistics as s

def plot_hist(generalDensity, category, bins=20):
    # set a grey background (use sns.set_theme() if seaborn version 0.11.0 or above) 
    sns.set(style="darkgrid")
    plt.title(category)
    sns.histplot(generalDensity, bins=bins,
                kde=True, kde_kws=dict(cut=3))
    # in the next version of the distplot function, one would have to write:
    # sns.distplot(data=df, x="sepal_length", kind='hist') # note that 'kind' is 'hist' by default
    plt.savefig(os.sep.join([folder,category+".png"]))
    plt.clf()
    #plt.show()

data = pd.read_csv("results.csv")
folder = "./histograms"

os.makedirs(folder,exist_ok=True)

unique_classes = list(set(data.classes))

to_delete = []

for unique_class in unique_classes:
    print(unique_class)

    class_data = data[data["classes"]==unique_class]

    distance = class_data.distance

    plot_hist(distance, unique_class)

    quartiles = s.quantiles(distance, n=5)
    qq = quartiles[1]-quartiles[0]
    
    pivot = 1000.0
    if quartiles[1] + qq >= quartiles[2]:
        if quartiles[2] + qq >= quartiles[3]:
            continue
        else:
            pivot = quartiles[3]
    else:
        pivot = quartiles[2]
    print("Pivot", pivot)
    asf = class_data[class_data["distance"] >= pivot]
    print(asf)

byInstances = data.distance

plot_hist(byInstances, "GENERAL-INSTANCES",30)

byClassMean = data.groupby('classes')['distance'].mean()

plot_hist(byClassMean, "GENERAL-CLASSES-MEAN", 7)

byClassVariance = data.groupby('classes')['distance'].var()

plot_hist(byClassVariance, "GENERAL-CLASSES-VARIANCE", 50)

byClassMedian = data.groupby('classes')['distance'].median()

plot_hist(byClassMedian, "GENERAL-CLASSES-MEDIAN", 20)