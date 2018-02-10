from os import listdir
from sys import argv
from scipy.stats.stats import pearsonr
import json
import numpy as np
import pandas as pd

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


corr_df = pd.DataFrame(columns=[
    'Meta Feature A',
    'Meta Feature B',
    'Pearson Coefficient',
    '2-tailed p-value'])

name_label = [
    "Average Size",
    "Standard Deviation of Size",
    "Average Density",
    "Standard Deviation of Density",
    meta_names[2]]

for i, row in enumerate(meta_features):
    for j, feature in enumerate(row):
        if not np.isfinite(feature):
            print(i, j)


for i in range(len(meta_features)):
    for j in range(len(meta_features)):
        x = np.array(meta_features[i])
        y = np.array(meta_features[j])
        a = [name_label[i], name_label[j]] + list(
            pearsonr(x, y))
        corr_df.loc[i * len(meta_features) + j] = a

corr_df.to_html('Info_metas_correlation.html')
corr_df
