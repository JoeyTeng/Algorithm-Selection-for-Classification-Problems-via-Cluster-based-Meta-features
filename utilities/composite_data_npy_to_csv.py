
# @Author: Joey Teng
# @Email:  joey.teng.dev@gmail.com
# @Filename: composite_data_npy_to_csv.py
# @Last modified by:   Joey Teng
# @Last modified time: 11-Apr-2018

import csv
import os
import sys

import numpy

CLUSTER_FEATURES = [
    "Average Size",
    "Standard Deviation of Size",
    "Average of Natural Logarithm of Inverse of Density",
    "Standard Deviation of Natural Logarithm of Inverse of Density",
    "Number of Clusters"]
META_FEATURE_NAMES = ['ClassEnt', 'AttrEnt', 'JointEnt', 'MutInfo',
                      'EquiAttr', 'NoiseRatio', 'StandardDev', 'Skewness',
                      'Kurtosis', 'treewidth', 'treeheight', 'NoNode',
                      'NoLeave', 'maxLevel', 'meanLevel', 'devLevel',
                      'ShortBranch', 'meanBranch', 'devBranch', 'maxAtt',
                      'minAtt', 'meanAtt', 'devAtt'] + CLUSTER_FEATURES


def main(path):
    # TODO:
    print(path, flush=True)
    files = ['{0}/{1}'.format(
        path.strip(), file[:-len('.cluster.npy')])
        for file in os.listdir(path)
        if file.find('.cluster.npy') != -1]
    files.sort()

    table = []
    for file in files:
        print("Loaded: {}".format(file), flush=True)
        row = [file[file.rfind('/') + 1:]] +\
            list(numpy.load("{}.npy".format(file))) +\
            list(numpy.load("{}.cluster.npy".format(file)))
        table.append(row)

    print("Writing into csv file...", flush=True)
    with open("{}/composited.csv".format(path), 'w', newline='') as csvfile:
        fieldnames = ["Dataset"] + META_FEATURE_NAMES
        writer = csv.writer(csvfile, dialect='excel')
        writer.writerow(fieldnames)
        writer.writerows(table)
    print("Completed.", flush=True)


if __name__ == '__main__':
    for path in sys.argv[1:]:
        main(path)
