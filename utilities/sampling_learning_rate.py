# @Author: Joey Teng
# @Email:  joey.teng.dev@gmail.com
# @Filename: sampling_learning_rate.py
# @Last modified by:   Joey Teng
# @Last modified time: 27-Apr-2018

import argparse
import collections
import copy
import json
import multiprocessing.pool
import os
import random


NUMBER_OF_PERCENTAGES = 10
NUMBER_OF_TEST_SETS = 10
NUMBER_OF_TRAINING_SETS = 10
PROCESS_COUNT = int(os.cpu_count() / 2)


def load_dataset(filename):
    return list(map(lambda x: x.strip(), open(filename, 'r').readlines()))


def generate_test_set(dataset):
    population = copy.deepcopy(dataset)
    random.shuffle(population)
    test_set = population[:len(population) // 10]
    remainder = population[len(population) // 10:]

    return test_set, remainder


def generate_training_sets(dataset, percentage, copies):
    training_sets = []
    for i in range(copies):
        population = copy.deepcopy(dataset)
        random.shuffle(population)
        training_sets.append(population[:len(population) * percentage // 100])
    return training_sets


def main(path):
    """main"""
    print("{} Start".format(path), flush=True)

    dataset = load_dataset(path)
    datasets = []
    for i in range(NUMBER_OF_TEST_SETS):
        sets = collections.defaultdict(list)
        sets['test set'], sets['remainder'] = generate_test_set(dataset)
        datasets.append(sets)

    json.dump(datasets, open("{}.learning_rate.json".format(path), 'w'))

    print("{} Complete".format(path), flush=True)


def traverse(paths):
    print("Starting Traverse Through", flush=True)
    files = []
    while paths:
        path = paths[0]
        paths = paths[1:]
        for file in os.listdir(path):
            if (file.endswith('.in')
                and file.find('.json') == -1
                and file.find('.log') == -1
                    and file.find('.DS_Store') == -1):
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
