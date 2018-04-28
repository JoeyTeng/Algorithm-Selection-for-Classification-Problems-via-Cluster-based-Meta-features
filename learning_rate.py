# @Author: Joey Teng
# @Email:  joey.teng.dev@gmail.com
# @Filename: learning_rate.py
# @Last modified by:   Joey Teng
# @Last modified time: 28-Apr-2018

import argparse
import collections
import copy
import json
import multiprocessing.pool
import os

import numpy
import sklearn.ensemble
import sklearn.tree

PROCESS_COUNT = int(os.cpu_count() / 2)


def split_data_target(dataset):
    return [[float(element) for element in row.strip().split(',')[:-1]] for row in dataset], [float(row.strip().split(',')[-1]) for row in dataset]


def generate_result(datasets, classifier, path):
    results = []
    for dataset in datasets:
        test_set = dataset['test set']
        result = collections.defaultdict(dict)
        for key, value in dataset.items():
            if (key == 'test set' or key == 'remainder'):
                continue
            if (key == '100'):
                value = [value[0]]
            else:
                try:
                    assert(sorted(value[0]) != sorted(value[1]))
                except AssertionError:
                    print(
                        "{} Warning: Repetition in training set with key {}".format(path, key))

            # print("DEBUG: Key {}, type {}".format(key, type(key)))

            print("{} Running on {}%".format(path, key), flush=True)
            result[key]['raw'] = []
            for training_set in value:
                clf = classifier()
                data, target = split_data_target(training_set)
                clf.fit(data, target)

                # print("{} Debug: Size of training set {}".format(path, len(data)))
                data, target = split_data_target(test_set)
                accuracy = clf.score(data, target)
                result[key]['raw'].append(accuracy)

            result[key]['average'] = numpy.average(result[key]['raw'])
            result[key]['standard deviation'] = numpy.std(result[key]['raw'])
            result[key]['range'] = max(
                result[key]['raw']) - min(result[key]['raw'])

        results.append(result)

    return results


def DecisionTreeClassifier():
    return sklearn.tree.DecisionTreeClassifier(min_impurity_split=1.0)


def RandomForestClassifier():
    return sklearn.ensemble.RandomForestClassifier(n_estimators=64)


def main(path):
    """main"""
    print("{} Start".format(path), flush=True)

    datasets = json.load(open(path, 'r'))
    results = {}
    print("{} Evaluating Random Forest".format(path), flush=True)
    results['Random Forest'] = generate_result(
        datasets, RandomForestClassifier, path)
    print("{} Evaluating Decision Tree".format(path), flush=True)
    results['Decision Tree'] = generate_result(
        datasets, DecisionTreeClassifier, path)

    json.dump(results, open(
        "{}.result.json".format(path[:-len('.json')]), 'w'))
    print("{} Complete".format(path), flush=True)


def traverse(paths):
    print("Starting Traverse Through", flush=True)
    files = []
    while paths:
        path = paths[0]
        paths = paths[1:]
        for file in os.listdir(path):
            if (file.endswith('.learning_rate.json')):
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


if __name__ == '__main__':
    paths = parse_path()

    pool = multiprocessing.pool.Pool(PROCESS_COUNT)
    list(pool.map(main, paths))
    pool.close()
    pool.join()
