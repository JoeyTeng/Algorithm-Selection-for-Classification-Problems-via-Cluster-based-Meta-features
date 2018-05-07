# @Author: Joey Teng
# @Email:  joey.teng.dev@gmail.com
# @Filename: artificial_datasets.py
# @Last modified by:   Joey Teng
# @Last modified time: 07-May-2018

import argparse
import bisect
import itertools
import json
import random

import numpy
import plotly

import download_png

POINTS = 30


class PlotGraph(object):
    @classmethod
    def __call__(cls, path, _data):
        return cls.run(path, _data)

    @classmethod
    def run(cls, path, _data, _layout):
        print("Plotting graph of: {}".format(path), flush=True)

        data = cls.plot_data_generation(_data)
        layout = cls.layout_generation()
        cls.plot(
            path,
            [data[1], data[2]],
            "scatter-linear",
            cls.layout(
                path,
                **layout['Linear'],
                **_layout))

        print("Graph Plotted: {}".format(path), flush=True)

    @classmethod
    def title_generation(cls, title, **kwargs):
        return "{}<br>{}".format(
            title,
            "".join(
                ["<br>{}: {}".format(key, value)
                 for key, value in kwargs.items()]))

    @classmethod
    def layout_generation(cls):
        return dict(
            Linear=dict(
                xaxis=dict(
                    type='linear'
                ),
                yaxis=dict(
                    type='linear'
                )
            )
        )

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
    def plot_offline(cls, fig, path, plot_type):
        filename = "{}.{}.html".format(path, plot_type)
        url = plotly.offline.plot(
            fig,
            image="png",
            image_filename="{}.{}".format(
                path[path.rfind('/') + 1:], plot_type),
            filename=filename,
            auto_open=False)

        destination = path[:path.rfind('/')]
        try:
            download_png.download(destination, url)
        except RuntimeError:
            print("RuntimeError occurs when downloading {}".format(url),
                  flush=True)
            return

        print("Offline Graph Plotted: {}.{}".format(
            path, plot_type), flush=True)

    @classmethod
    def layout(cls, title, xaxis=None, yaxis=None, **kwargs):
        layout = dict(
            title=cls.title_generation(title, **kwargs),
            xaxis=xaxis,
            yaxis=yaxis)

        return layout

    @classmethod
    def plot(cls, path, data, plot_type, layout):
        fig = plotly.graph_objs.Figure(data=data, layout=layout)
        cls.plot_offline(fig, path, plot_type)


def label(point, angles):
    x, y = point
    angle = (y - 0.5)
    angles = numpy.tan(angles) * (x - 0.5)
    angles = angles.tolist()
    angles.sort()
    count = bisect.bisect_left(angles, angle)

    if (count % 2):
        return 1
    return 0


def main(args):
    n = args.n  # number of linear separators
    randomise = args.random
    path = args.o

    if randomise:
        angles = numpy.array([random.random() for i in range(n)]) * numpy.pi
        angles = numpy.array(angles.tolist().sort())
    else:
        angles = numpy.array(list(range(n))) / n * numpy.pi

    points = [coordinate
              for coordinate in itertools.product(range(POINTS), repeat=2)]
    points = numpy.array(points)
    points = (points - 0) / (POINTS - 1 - 0)  # Normalization
    points = points.tolist()

    labeled_points = [(point[0], point[1], label(point, angles))
                      for point in points]

    with open(path, 'w') as output:
        output.writelines(['{}, {}, {}\n'.format(*point)
                           for point in labeled_points])

    return labeled_points


def plot(points, args):
    pass


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
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    points = main(args)
    plot(points, args)
