"""Calculate data for learning rate calculation.

Taking data from sampling_learning_rate.py
Output may be processed by plot_learning_rate.py
"""
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
    """Split the input CSV files into X, y vectors for sklearn implementations.

    Args:
        dataset (list): List of list of floats.
            [
                [0...n - 1]: X, feature vector
                [-1]: y, label
            ]

    Returns:
        tuple: (X, y) for sklearn implementations

    """
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
    """Resample from separated training sets to generate smaller training sets.

    No instance will present in one new training set more than once.
    Mechanism is to shuffle, then pick the first percentage% instances.

    Args:
        dataset (list): List of vectors (features + label)
        percentage (number that supports __mul__ and __floordiv__):
            This decides the size of new training set generated related to the
            population.
        copies (int): The number of new training sets required.

    Returns:
        list: list of new training datasets
            list of list of vectors

    """
    training_sets = []
    i = copies
    while i > 0:
        population = copy.deepcopy(dataset)
        random.shuffle(population)
        training_sets.append(population[:len(population) * percentage // 100])
        i -= 1

    return training_sets


def generate_result(datasets, classifier, path):
    """Generate the learning rate accuracies.

    Args:
        datasets (dict): {
            'test set': testing set for the specific dataset
            'remainder': instances in the dataset but not testing set
        }
        classifier (func): a function that will return an instance of
            sklearn classifier.
        path (str): path of the dataset, for logging only.

    Returns:
        dict: dict of dict {
            percentage: results under respective portion of training data {
                'raw' (list): raw accuracy values [
                    accuracy values of each training set-testing set pairs
                ]
                'average': average of 'raw'
                'standard deviation': standard deviation of 'raw'
                'range': range of 'raw'
            }
        }

    """
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

            print("{} Running on {}%. Testing set count: {}/{}".format(
                path, percentage, (len(results) + 1), len(datasets)),
                flush=True)
            result[percentage]['raw'] = []
            for training_set in value:
                clf = classifier()
                data, target = split_data_target(training_set)

                if data:
                    clf.fit(data, target)
                    data, target = split_data_target(test_set)
                    accuracy = clf.score(data, target)
                else:
                    accuracy = 0.0

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
    """Wrap a default Random Forest classifier with fixed parameter."""
    return sklearn.ensemble.RandomForestClassifier(n_estimators=64)


def main(path):
    """Start main function here.

    Run tasks and dump result files.
    """
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
    """Travsere to append all files in children folders into the task queue.

    Args:
        paths (list): Paths of all folders to be detected

    Returns:
        list: Paths of all files added in the task queue

    """
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
    """Parse the arguments.

    No argument is required for calling this function.

    Returns:
        Namespace: parsed arguments enclosed by an object defined in argparse

    """
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
