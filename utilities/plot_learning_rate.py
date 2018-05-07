# @Author: Joey Teng
# @Email:  joey.teng.dev@gmail.com
# @Filename: plot_learning_rate.py
# @Last modified by:   Joey Teng
# @Last modified time: 28-Apr-2018

import argparse
import collections
import copy
import json
import multiprocessing
import os

import numpy
import plotly
import scipy.optimize
import scipy.stats

import download_png


PROCESS_COUNT = int(os.cpu_count() / 2)


class PlotGraph(object):
    origins = 0

    @classmethod
    def __call__(cls, path, _data):
        return cls.run(path, _data)

    @classmethod
    def run(cls, path, _data):
        print("Plotting graph of: {}".format(path), flush=True)

        _data['x'].extend([0] * cls.origins)
        _data['y'].extend([0] * cls.origins)
        _data['max'].extend([0] * cls.origins)
        _data['min'].extend([0] * cls.origins)

        coefficient, formula, fit_x, fit_y, predicted_y = cls.lsq_logistic_fit(
            _data['x'], _data['y'])
        data = dict(
            x=_data['x'][:-cls.origins] or _data['x'],
            y=_data['y'][:-cls.origins] or _data['y'],
            max=_data['max'][:-cls.origins] or _data['max'],
            min=_data['min'][:-cls.origins] or _data['min'])

        pearsonr_y = predicted_y
        y = _data['y']

        pearsonr = scipy.stats.pearsonr(y, pearsonr_y)
        r_square = pearsonr[0] ** 2
        data = cls.plot_data_generation(data, fit_x, fit_y)
        layout = cls.layout_generation()
        cls.plot(
            path,
            [data[1], data[2]],
            "scatter-linear",
            cls.layout(
                path,
                **layout['Linear'],
                formula=formula,
                r_square=r_square))
        # cls.plot(
        #     path,
        #     [data[1], data[2]],
        #     "scatter-logarithmic",
        #     cls.layout(
        #         path,
        #         **layout['Logarithmic'],
        #         formula=formula,
        #         r_square=r_square))

        print("Graph Plotted: {}".format(path), flush=True)

        return dict(coefficients=coefficient,
                    r_square=r_square)

    @classmethod
    def title_generation(cls, path, **kwargs):
        return "{}".format(
            path[path.rfind('/') + 1:].replace(
                '.learning_rate.result.json.', ' ') + "".join(
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
            ),
            Logarithmic=dict(
                xaxis=dict(
                    type='linear',
                    range=[0, 100]
                ),
                yaxis=dict(
                    type='log',
                    range=[-1, 0]
                )
            )
        )

    @classmethod
    def plot_data_generation(cls, _data, fit_x, fit_y):
        return [
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
                ),
                name='Average/Range of averages over 10 test sets'
            ),
            plotly.graph_objs.Scatter(
                x=_data['x'],
                y=_data['y'],
                mode='markers',
                name='Averages over 10 test sets'
            ),
            plotly.graph_objs.Scatter(
                x=fit_x,
                y=fit_y,
                mode='lines',
                name='BFL')
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
    def layout(cls, path, xaxis=None, yaxis=None, **kwargs):
        layout = dict(
            title=cls.title_generation(path, **kwargs),
            xaxis=xaxis,
            yaxis=yaxis)

        return layout

    @classmethod
    def plot(cls, path, data, plot_type, layout):
        fig = plotly.graph_objs.Figure(data=data, layout=layout)
        cls.plot_offline(fig, path, plot_type)

    @classmethod
    def bar(cls, path, data, **kwargs):
        layout = dict(
            title=cls.title_generation(path, **kwargs))
        plotly.offline.plot(
            plotly.graph_objs.Figure(data=data, layout=layout),
            image="png",
            image_filename="{}.error_bar.html".format(
                path[path.rfind('/') + 1:]),
            filename="{}.error_bar.html".format(path))

    @classmethod
    def scatter(cls, path, data, **kwargs):
        layout = dict(
            title=cls.title_generation(path, **kwargs))
        plotly.offline.plot(
            plotly.graph_objs.Figure(data=data, layout=layout),
            image="png",
            image_filename="{}.scatter.html".format(
                path[path.rfind('/') + 1:]),
            filename="{}.scatter.html".format(path))

    @staticmethod
    def exponenial_func(x, a, b, c):
        return a * numpy.exp(-b * x) + c

    @classmethod
    def exp_fit(cls, _x, _y):
        y = numpy.array(_y)
        x = numpy.array(_x)
        a, b, c = scipy.optimize.curve_fit(cls.exponenial_func, x, y)[0]
        __x = numpy.array(list(range(len(x) * 2)))
        __x = __x / (max(__x) - min(__x)) * (max(x) - min(x)) + min(x)
        __y = cls.exponenial_func(__x, a, b, c)
        predicted_y = cls.exponenial_func(x, a, b, c)

        return (a, b, c), "y = ({}) * e^(({}) * x) + ({})".format(
            a, b, c), __x.tolist(), __y.tolist(), predicted_y.tolist()

    @classmethod
    def lsq_exp_fit(cls, _x, _y):
        y = 1 - numpy.array(_y)
        y = numpy.log(y)
        x = numpy.array(_x)
        k, e = numpy.polyfit(x, y, 1)
        __x = numpy.array(list(range(len(_x) * 2)))
        __x = __x / (max(__x) - min(__x)) * (max(x) - min(x)) + min(x)
        __y = 1 - numpy.e ** (k * __x + e)
        predicted_y = 1 - numpy.e ** (k * x + e)

        return (k, e), "y = 1 - e^(({})x + ({}))".format(
            k, e), __x.tolist(), __y.tolist(), predicted_y.tolist()

    @classmethod
    def lsq_ln_fit(cls, _x, _y):
        y = numpy.log(_y)
        x = numpy.array(_x)
        k, c = numpy.polyfit(x, y, 1)
        __x = numpy.array(list(range(len(_x) * 2)))
        __x = __x / (max(__x) - min(__x)) * (max(x) - min(x)) + min(x)
        __y = numpy.exp(k * __x + c)
        predicted_y = numpy.exp(k * x + c)

        return (k, c), "y = e^(({})x + ({}))".format(
            k, c), __x.tolist(), __y.tolist(), predicted_y.tolist()

    @staticmethod
    def logistic_linearisation(y):
        if y == 'formula':
            return "ln(y^(-1) - 1) = kx + c"
        y = numpy.array(y)
        return numpy.log(y**(-1) - 1)

    @classmethod
    def lsq_logistic_fit(cls, _x, _y):
        y = cls.logistic_linearisation(_y)
        x = numpy.array(_x)
        k, c = numpy.polyfit(x, y, 1)
        __x = numpy.array(list(range(len(_x) * 2)))
        __x = __x / (max(__x) - min(__x)) * (max(x) - min(x)) + min(x)
        __y = 1 / (1 + numpy.exp(k * __x + c))
        predicted_y = 1 / (1 + numpy.exp(k * x + c))

        return (k, c), "y = 1 / (1 + e^(({})x + ({})))".format(
            k, c), __x.tolist(), __y.tolist(), predicted_y.tolist()


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

    return PlotGraph()(name, _data)


def main(path):
    """main"""

    printing_data = {}
    results = json.load(open(path, 'r',))
    for key, value in results.items():
        printing_data[key] = plot("{}.{}".format(
            path, key.replace(' ', '_')), value)

    return (path[path.rfind('/') + 1:].replace(
        '.learning_rate.result.json', ''),
        printing_data)


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
        description="Plot graphs for learning rate & Best Fit Line.\n Use plotly & selenium with headless Firefox")
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
    printing_data = pool.map(main, paths)
    pool.close()
    pool.join()

    path = "{}/_values.json".format(paths[0][:paths[0].rfind('/')])
    print("Dumping results into {}...".format(path), flush=True)
    print(printing_data, flush=True)
    with open(path, 'w') as f:
        json.dump(printing_data, f)


if __name__ == '__main__':
    paths = parse_path()

    multiprocess(paths)

    print("Program Ended", flush=True)
