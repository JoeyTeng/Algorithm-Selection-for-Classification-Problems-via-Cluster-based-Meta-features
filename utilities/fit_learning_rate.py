# @Author: Joey Teng
# @Email:  joey.teng.dev@gmail.com
# @Filename: fit_learning_rate.py
# @Last modified by:   Joey Teng
# @Last modified time: 28-Apr-2018


import argparse
import collections
import copy
import json
# import multiprocessing.pool
import os

import numpy
import plotly


PROCESS_COUNT = int(os.cpu_count() / 2)


class GraphPlotter(type):
    counter = 0
    Threshold = 10
    lock = False

    def __call__(cls, path, _data):
        while not cls.lock:
            cls.lock = True
        if cls.counter >= cls.Threshold:
            input("Press enter to continue...")
            cls.counter = 0

        data = [
            plotly.graph_objs.Scatter(
                x=_data['x'],
                y=_data['y'],
                error_y=dict(
                    type='data',
                    symmetric=False,
                    array=[_data['max'][i] - _data['y'][i]
                           for i in range(len(_data['y']))],
                    arrayminus=[_data['y'][i] - _data['min'][i]
                                for i in range(len(_data['y']))]
                )
            )
        ]
        fig = plotly.graph_objs.Figure(data=data)
        plotly.offline.plot(
            fig,
            image="png",
            image_filename=path[path.rfind('/') + 1:],
            filename=path)

        cls.counter += 1
        cls.lock = False


class PlotGraph(metaclass=GraphPlotter):
    pass


def plot(name, data):
    _data = collections.defaultdict(list)
    for key in data[0].keys():
        raw = []
        percentage = int(key)
        _data['x'].append(percentage)
        for fold in range(len(data)):
            raw.append(data[fold][key]['average'])
        _data['raw'].append(raw)
        _data['y'].append(numpy.average(raw))
        _data['max'].append(max(raw))
        _data['min'].append(min(raw))
    PlotGraph(name, _data)


def main(path):
    """main"""
    results = json.load(open(path, 'r',))
    for key, value in results.items():
        plot("{}.{}".format(path, key.replace(' ', '_')), value)


def traverse(paths):
    print("Starting Traverse Through", flush=True)
    files = []
    while paths:
        path = paths[0]
        paths = paths[1:]
        for file in os.listdir(path):
            if file.endswith('.result.json'):
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

    # pool = multiprocessing.pool.Pool(PROCESS_COUNT)
    # list(pool.map(main, paths))
    # pool.close()
    # pool.join()
    list(map(main, paths))
