# @Author: Joey Teng
# @Email:  joey.teng.dev@gmail.com
# @Filename: artificial_datasets.py
# @Last modified by:   Joey Teng
# @Last modified time: 19-Jul-2018

import argparse
import bisect
import collections
import itertools
import json
import os
import random

import numpy
import plotly
import sklearn.neighbors

import download_png


INCREMENT = dict(
    # -x, -y, angle/, random-angle, +n for uniform
    corner=(0, 0, 2, 0, 1),
    side=(),
    centre=(0.5, 0.5, 1, 0.5, 0)
)


class PlotGraph(object):
    @classmethod
    def __call__(cls, *args, **kwargs):
        return cls.run(*args, **kwargs)

    @classmethod
    def run(cls, path, _data, _layout):
        print("Plotting graph of: {}".format(path), flush=True)

        data = cls.plot_data_generation(_data)
        layout = cls.layout(
            "2-D Artificial Dataset",
            **_layout)
        cls.plot(
            path,
            data,
            layout)

        print("Graph Plotted: {}".format(path), flush=True)

    @classmethod
    def title_generation(cls, title, **kwargs):
        return "{}{}".format(
            title,
            "".join(
                ["<br>{}: {}".format(key, value)
                 for key, value in kwargs.items()]))

    @classmethod
    def plot_data_generation(cls, _data):
        return [
            plotly.graph_objs.Scatter(
                x=_data[0]['x'],
                y=_data[0]['y'],
                mode='markers',
                name='category 0'
            ),
            plotly.graph_objs.Scatter(
                x=_data[1]['x'],
                y=_data[1]['y'],
                mode='markers',
                name='category 1'
            )
        ]

    @classmethod
    def plot_offline(cls, fig, path):
        filename = "{}.html".format(path[:-len('.png')])
        url = plotly.offline.plot(
            fig,
            image="png",
            image_filename=path[path.rfind('/') + 1:-len('.png')],
            filename=filename,
            auto_open=False)

        destination = path[:path.rfind('/')]
        try:
            download_png.download(destination, url)
        except RuntimeError:
            print("RuntimeError occurs when downloading {}".format(url),
                  flush=True)
            return

        print("Offline Graph Plotted: {}".format(path), flush=True)

    @classmethod
    def layout(cls, title, **kwargs):
        layout = dict(
            title=cls.title_generation(title, **kwargs))

        return layout

    @classmethod
    def plot(cls, path, data, layout):
        fig = plotly.graph_objs.Figure(data=data, layout=layout)
        cls.plot_offline(fig, path)


def label(point, separators):
    count = 0
    for separator in separators:
        matrix = numpy.matrix([
            numpy.array(point) - numpy.array(separator[0]),
            numpy.array(separator[1]) - numpy.array(separator[0])])
        if numpy.linalg.det(matrix) < 0:
            count += 1

    if (count % 2):
        return 1
    return 0


def intersection(args):
    n = args.n  # number of linear separators
    randomise = args.random
    increment = INCREMENT[args.intersection]

    if randomise:
        angles = numpy.array([random.random() for i in range(n)])
        angles = (angles - increment[3]) * numpy.pi / increment[2]
        angles = numpy.array(sorted(angles.tolist()))
    else:
        angles = ((numpy.array(list(range(n))) + 1) / (n + increment[4])
                  * numpy.pi / increment[2])

    separators = []
    for angle in angles:
        separators.append((
            (increment[0], increment[1]), (
                numpy.cos(angle) + increment[0],
                numpy.sin(angle) + increment[1])))

    return separators


def orthogonal(args):
    n = args.n  # number of linear separators
    randomise = args.random

    if args.nh == -1 and args.nv != -1:
        n_v = args.nv
        n_h = n - n_v
    elif args.nh != -1 and args.nv == -1:
        n_h = args.nh
        n_v = n - n_h
    elif args.nh != -1 and args.nv != -1:
        n_h = args.nh
        n_v = args.nv
    else:
        n_h = n // 2
        n_v = n - n_h

    if randomise:
        distance = [random.random() for i in range(n)]
        horizontal = distance[n_v:][:n_h]
        vertical = distance[:n_v]
    else:
        horizontal = (numpy.array(list(range(n_h))) + 1) / (n_h + 1)
        vertical = (numpy.array(list(range(n_v))) + 1) / (n_v + 1)

    separators = [(
        (0.0, y),
        (1.0, y)
    ) for y in horizontal] + [(
        (x, 0.0),
        (x, 1.0)
    ) for x in vertical]

    return separators


def kNN(args):
    n = args.n  # number of centroids
    # Class h && class v
    if args.nh == -1 and args.nv != -1:
        n_v = args.nv
        n_h = n - n_v
    elif args.nh != -1 and args.nv == -1:
        n_h = args.nh
        n_v = n - n_h
    elif args.nh != -1 and args.nv != -1:
        n_h = args.nh
        n_v = args.nv
    else:
        n_h = n // 2
        n_v = n - n_h

    return ([(random.random(), random.random(), 0) for i in range(n_h)] +
            [(random.random(), random.random(), 1) for i in range(n_v)])


class kNN_predict(object):
    predictor = None

    @classmethod
    def __call__(cls, point, centroids, initialise=False):
        if initialise or not cls.predictor:
            cls.predictor = sklearn.neighbors.KNeighborsClassifier(
                n_neighbors=len(centroids),
                weights='distance')
            X = numpy.array([[p[0], p[1]] for p in centroids])
            y = numpy.array([p[2] for p in centroids])
            cls.predictor.fit(X, y)
        return cls.predictor.predict(numpy.array([point]))[0]


def main(args):
    path = args.o
    number_of_points = int((args.np) ** 0.5)
    mode = (
        + (int(args.intersection != ''))
        + (int(args.orthogonal) << 1)
        + (int(args.kNN) << 2))

    if mode == 0:
        print("Please choose only one mode!")
        return None
    elif (mode != 1 and
          mode != (1 << 1) and
          mode != (1 << 2)):
        print("Please choose any mode. -h to check details")
        return None

    points = [coordinate
              for coordinate in itertools.product(
                  range(number_of_points), repeat=2)]
    points = numpy.array(points)
    points = (points - 0) / (number_of_points - 1 - 0)  # Normalization
    points = points.tolist()

    if args.kNN:
        centroids = kNN(args)
        labeled_points = [(point[0], point[1], kNN_predict()(point, centroids))
                          for point in points]
        json.dump(centroids, open("{}.centroids.json".format(path), 'w'))
    else:
        if args.intersection:
            separators = intersection(args)
        elif args.orthogonal:
            separators = orthogonal(args)

        labeled_points = [(point[0], point[1], label(point, separators))
                          for point in points]
        json.dump(separators, open("{}.separators.json".format(path), 'w'))

    with open(path, 'w') as output:
        output.writelines(['{}, {}, {}\n'.format(*point)
                           for point in labeled_points])
    return labeled_points


def plot(points, args):
    n = args.n  # number of linear separators
    randomise = args.random
    path = args.save_image_to
    if (not path.startswith('/')):  # Using relative path instead of absolute
        path = '{}/{}'.format(os.getcwd(), path)

    data = [collections.defaultdict(list),
            collections.defaultdict(list)]
    for point in points:
        data[point[2]]['x'].append(point[0])
        data[point[2]]['y'].append(point[1])

    additional_info = dict(
        number_of_separators=n,
        randomised_angles=randomise,
        number_of_points=len(points),
        ratio_of_zero_to_all=len(data[0]['x']) / len(points)
    )

    PlotGraph()(path, data, additional_info)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate 2-D Artificial Datasets using linear separators"
    )
    parser.add_argument('-n', action='store', type=int, default=0,
                        help='The number of linear separators in the dataset')
    parser.add_argument('--random', action='store_true',
                        help=' '.join(['state if you want to use randomised',
                                       'angles (interval) for separators']))
    parser.add_argument('-o', action='store', type=str, default='data.out',
                        help='Path to where the generated dataset is stored')
    parser.add_argument('--save_image_to', action='store', type=str,
                        default="{}/data.png".format(os.getcwd()),
                        help='Path to where the graph plotted is stored')
    parser.add_argument('-np', action='store', type=int,
                        default=900,  # A random choice though
                        help='The number of data instance you want')
    parser.add_argument('--intersection', action='store',
                        choices=['corner', 'side', 'centre'],
                        default='',
                        help=''.join([
                            'Use tilted separators. This indicates the point',
                            'of intersection of separators. Default: \'\'']))
    parser.add_argument('--orthogonal', action='store_true',
                        help='Use orthogonal separators instead.')
    parser.add_argument('-nh', action='store', type=int, default=-1,
                        help=' '.join([
                            'The number of horizontal linear separators',
                            'in the dataset for orthogonal mode only']))
    parser.add_argument('-nv', action='store', type=int, default=-1,
                        help=' '.join([
                            'The number of vertical linear separators',
                            'in the dataset for orthogonal mode only']))
    parser.add_argument('--kNN', action='store_true',
                        help=' '.join([
                            'Use full-NN based method to assign the class.',
                            'Assume --random by default.']))
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    points = main(args)
    if points:
        plot(points, args)
