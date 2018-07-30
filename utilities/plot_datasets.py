# @Author: Joey Teng
# @Email:  joey.teng.dev@gmail.com
# @Filename: plot_datasets.py
# @Last modified by:   Joey Teng
# @Last modified time: 31-Jul-2018

import argparse
import collections
import os

import plotly

import download_png


class PlotGraph(object):
    @classmethod
    def __call__(cls, *args, **kwargs):
        return cls.run(*args, **kwargs)

    @classmethod
    def run(cls, path, _data):
        print("Plotting graph of: {}".format(path), flush=True)

        data = cls.plot_data_generation(_data)
        cls.plot(
            path,
            data)

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
                marker=dict(
                    symbol='circle'
                ),
                name='category 0'
            ),
            plotly.graph_objs.Scatter(
                x=_data[1]['x'],
                y=_data[1]['y'],
                mode='markers',
                marker=dict(
                    symbol='x'
                ),
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
    def plot(cls, path, data):
        fig = plotly.graph_objs.Figure(data=data)
        cls.plot_offline(fig, path)


def load(args):
    path = args.i

    with open(path, 'r') as reader:
        labeled_points = reader.readlines()
        labeled_points = list(map(
            lambda row: tuple(
                (lambda cell:
                    (float(cell[0]), float(cell[1]), int(cell[2])))
                (row.split(','))),
            labeled_points))
    return labeled_points


def plot(points, args):
    path = args.save_image_to
    if (not path.startswith('/')):  # Using relative path instead of absolute
        path = '{}/{}'.format(os.getcwd(), path)

    data = [collections.defaultdict(list),
            collections.defaultdict(list)]
    for point in points:
        data[point[2]]['x'].append(point[0])
        data[point[2]]['y'].append(point[1])

    PlotGraph()(path, data)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot graphs for 2-D Binary Datasets"
    )
    parser.add_argument('-i', action='store', type=str, default='data.in',
                        help='Path to where the generated dataset is stored')
    parser.add_argument('--save_image_to', action='store', type=str,
                        default="{}/data.png".format(os.getcwd()),
                        help='Path to where the graph plotted is stored')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    points = load(args)
    if points:
        plot(points, args)
