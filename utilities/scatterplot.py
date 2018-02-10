from matplotlib import pyplot as plt
from os import listdir
from sys import argv
import json

meta_names = [
    'Size versus Number of Clusters',
    'Density distribution over 10 intervals',
    'Number of Clusters']

files = []
for path in argv[1:]:
    files.extend([
        "{0}/{1}".format(path, filepath)
        for filepath in listdir(path) if filepath.endswith('.output.json')])

meta_features_by_dataset = []
for filename in files:
    meta_features_by_dataset.append(json.load(open(filename, 'r')))

meta_features = [[] for i in range(len(meta_names) + 2)]
for metas_by_dataset in meta_features_by_dataset:
    for i in range(len(meta_names) - 1):
        meta_features[i * 2].append(
            metas_by_dataset[meta_names[i]]['_population']['average'])  # /
        # metas_by_dataset[meta_names[i]]['_population']['range'])
        meta_features[i * 2 + 1].append(
            metas_by_dataset[meta_names[i]]
            ['_population']['standard deviation'])  # /
        # metas_by_dataset[meta_names[i]]['_population']['range'])
    meta_features[-1].append(metas_by_dataset[meta_names[-1]]['_population'])

name_label = [
    "Average Size",
    "Standard Deviation of Size",
    "Average Density",
    "Standard Deviation of Density",
    meta_names[2]]

fig1 = plt.figure(figsize=[25, 25])
fig1.clear()

for i in range(len(meta_features)):
    for j in range(len(meta_features)):
        if i != j:
            x = meta_features[i]
            namex = name_label[i]
            y = meta_features[j]
            namey = name_label[j]
            sub = fig1.add_subplot(5, 5, i * 5 + j + 1)
            sub.scatter(x, y)  # , c="specs")
            sub.set_xlabel(namex)
            sub.set_ylabel(namey)
fig1.savefig('Meta_Features_Correlations.png')

plt.show(fig1)
