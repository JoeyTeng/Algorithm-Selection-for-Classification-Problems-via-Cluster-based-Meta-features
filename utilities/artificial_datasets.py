# @Author: Joey Teng
# @Email:  joey.teng.dev@gmail.com
# @Filename: artificial_datasets.py
# @Last modified by:   Joey Teng
# @Last modified time: 07-May-2018

import argparse
import bisect
import collections
import itertools
import json
import os
import random

import numpy
import plotly

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


def label(point, angles, increment):
    x, y = point
    point_y = (y - increment[1])
    comparing_y = numpy.tan(angles) * (x - increment[0])
    comparing_y = comparing_y.tolist()
    comparing_y.sort()
    count = bisect.bisect_left(comparing_y, point_y)

    if (count % 2):
        return 1
    return 0


def main(args):
    n = args.n  # number of linear separators
    randomise = args.random
    path = args.o
    number_of_points = int((args.np) ** 0.5)
    increment = INCREMENT[args.intersection]

    if randomise:
        angles = numpy.array([random.random() for i in range(n)])
        angles = (angles - increment[3]) * numpy.pi / increment[2]
        angles = numpy.array(sorted(angles.tolist()))
    else:
        angles = ((numpy.array(list(range(n))) + 1) / (n + increment[4])
                  * numpy.pi / increment[2])

    points = [coordinate
              for coordinate in itertools.product(
                  range(number_of_points), repeat=2)]
    points = numpy.array(points)
    points = (points - 0) / (number_of_points - 1 - 0)  # Normalization
    points = points.tolist()

    labeled_points = [(point[0], point[1], label(point, angles, increment))
                      for point in points]

    with open(path, 'w') as output:
        output.writelines(['{}, {}, {}\n'.format(*point)
                           for point in labeled_points])
    json.dump(
        (angles / numpy.pi).tolist(),
        open("{}.angles.json".format(path), 'w'))
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
        ratio_of_zero_to_all=len(data[0]) / len(points)
    )

    PlotGraph()(path, data, additional_info)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate 2-D Artificial Datasets using linear separators"
    )
    parser.add_argument('-n', action='store', type=int, default=0,
                        help='The number of linear separators in the dataset')
    parser.add_argument('--random', action='store_true',
                        help=''.join(['state if you want to use randomised',
                                      'angles (interval) for separators']))
    parser.add_argument('-o', action='store', type=str, default='data.out',
                        help='Path to where the generated dataset is stored')
    parser.add_argument('--save_image_to', action='store', type=str,
                        default="{}/data.png".format(os.getcwd()),
                        help='Path to where the graph plotted is stored')
    parser.add_argument('-np', action='store', type=int,
                        default=900,  # A random choice though
                        help='The number of data instance you want')
    parser.add_argument('intersection', action='store',
                        choices=['corner', 'side', 'centre'],
                        help='Point of intersection of separators')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    points = main(args)
    plot(points, args)
