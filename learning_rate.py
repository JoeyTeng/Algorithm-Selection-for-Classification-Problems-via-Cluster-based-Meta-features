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
import random

import numpy
import sklearn.ensemble
import sklearn.tree

NUMBER_OF_PERCENTAGES = 100
NUMBER_OF_TRAINING_SETS = 10  # Folds
PROCESS_COUNT = int(os.cpu_count() / 2)


def split_data_target(dataset):
    try:
        return ([[float(element)
                  for element in row.strip().split(',')[:-1]]
                 for row in dataset],
                [float(row.strip().split(',')[-1])
                 for row in dataset])
    except ValueError:
        print("dataset {}".format(dataset))
        raise ValueError


def generate_training_sets(dataset, percentage, copies):
    training_sets = []
    for i in range(copies):
        population = copy.deepcopy(dataset)
        random.shuffle(population)
        training_sets.append(population[:len(population) * percentage // 100])
    return training_sets


def generate_result(datasets, classifier, path):
    results = []
    for dataset in datasets:
        test_set = dataset['test set']
        result = collections.defaultdict(dict)
        for percentage in range(
                100 // NUMBER_OF_PERCENTAGES,
                100 + 100 // NUMBER_OF_PERCENTAGES,
                100 // NUMBER_OF_PERCENTAGES):

            if (percentage == 100):
                value = [copy.deepcopy(dataset['remainder'])]
            else:
                value = generate_training_sets(
                    dataset['remainder'],
                    percentage,
                    NUMBER_OF_TRAINING_SETS)

            print("{} Running on {}%".format(path, percentage), flush=True)
            result[percentage]['raw'] = []
            for training_set in value:
                clf = classifier()
                data, target = split_data_target(training_set)
                clf.fit(data, target)

                data, target = split_data_target(test_set)
                accuracy = clf.score(data, target)
                result[percentage]['raw'].append(accuracy)

            result[percentage]['average'] = numpy.average(
                result[percentage]['raw'])
            result[percentage]['standard deviation'] = numpy.std(
                result[percentage]['raw'])
            result[percentage]['range'] = max(
                result[percentage]['raw']) - min(result[percentage]['raw'])

        results.append(result)

    return results


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
