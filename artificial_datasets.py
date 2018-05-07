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

POINTS = 30


def label(point, angles):
    x, y = point
    angle = (y - 0.5)
    angles = numpy.tan(angles) * (x - 0.5)
    angles = angles.tolist()
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
    main(args)
