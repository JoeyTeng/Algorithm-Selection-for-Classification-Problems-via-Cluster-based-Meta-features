# @Author: Joey Teng
# @Email:  joey.teng.dev@gmail.com
# @Filename: extract_meta_features.py
# @Last modified by:   Joey Teng
# @Last modified time: 07-May-2018

# @Author: Joey Teng
# @Email:  joey.teng.dev@gmail.com
# @Filename: preprocessing.py
# @Last modified by:   Joey Teng
# @Last modified time: 09-Feb-2018

import argparse
import collections
import json
import multiprocessing
import os

import numpy

PROCESS_COUNT = int(os.cpu_count() / 2)
rel_tol = 1e-09

meta_names = [
    'Number of Clusters',
    'Ratio of Number of Category 0 Clusters versus Number of Clusters',
    'Normalized radius in 10 intervals',
    'Normalized size in 10 intervals']


def main(path):
    """main

    Read raw data file: .clusters.json
    """
    print(path, flush=True)

    metas = json.load(open(path, 'r'))
    population = len(metas['0']) + len(metas['1'])

    meta_features = []
    meta_features.append(
        population)
    meta_features.append(
        len(metas['0']) / population)

    raw_metas = collections.defaultdict(list)
    for value in metas['0'] + metas['1']:
        raw_metas['radius'].append(value['radius'])
        raw_metas['size'].append(value['size'])

    radii = numpy.array(raw_metas['radius'])
    sizes = numpy.array(raw_metas['size'])
    # Normalization
    radii_range = max(max(radii) - min(radii), rel_tol)
    radii = (radii - min(radii)) / radii_range
    sizes_range = max(max(sizes) - min(sizes), rel_tol)
    sizes = (sizes - min(sizes)) / sizes_range

    offset = len(meta_features)
    meta_features.extend([0 for i in range(20)])

    for radius in radii:
        meta_features[offset + int(radius * 9)] += 1
    for size in sizes:
        meta_features[offset + 10 + int(size * 9)] += 1

    numpy.save(
        "{0}.output".format(path[:-len('.clusters.json')]),
        numpy.array(meta_features))
    json.dump(
        meta_features,
        open("{0}.output.json".format(path[:-len('.clusters.json')]), 'w'))


def traverse(paths):
    print("Starting Traverse Through", flush=True)
    files = []
    while paths:
        path = paths[0]
        paths = paths[1:]
        for file in os.listdir(path):
            if file.endswith('.clusters.json'):
                files.append('{0}/{1}'.format(path, file))
            elif os.path.isdir('{0}/{1}'.format(path, file)):
                paths.append('{0}/{1}'.format(path, file))

    print("Traverse Completed.", flush=True)
    return files


def parse_path():
    parser = argparse.ArgumentParser(
        description="Generate Datasets for Detecting Learning Rate")
    parser.add_argument('-r', action='store', nargs='+', default=[],
                        help='Recursively processing all files in the folder')
    parser.add_argument('-i', action='store', nargs='+', default=[],
                        help='Files that need to be processed')
    args = parser.parse_args()
    paths = []
    if (args.r):
        paths = traverse(args.r)
    paths.extend(args.i)
    paths.sort()

    return paths


def multiprocess(paths):
    pool = multiprocessing.Pool(
        PROCESS_COUNT)
    print("Mapping tasks...", flush=True)
    pool.map(main, paths)
    pool.close()
    pool.join()


if __name__ == '__main__':
    paths = parse_path()

    multiprocess(paths)

    print("Program Ended", flush=True)
