# @Author: Joey Teng
# @Email:  joey.teng.dev@gmail.com
# @Filename: paired_t_test.py
# @Last modified by:   Joey Teng
# @Last modified time: 31-Jul-2018
import argparse
import json

import numpy
import scipy.stats


def main(args):
    pathA, pathB = args.i
    print(pathA, pathB, flush=True)

    dataA = numpy.matrix(json.load(open(pathA))).T.tolist()
    dataB = numpy.matrix(json.load(open(pathB))).T.tolist()

    p_values = []
    for index in range(len(dataA)):
        output = scipy.stats.ttest_rel(dataA[index], dataB[index])
        p_values.append(output[1])

    print(numpy.matrix(dataA) - numpy.matrix(dataB))
    print(p_values)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calculate Pearson r square value pair-wisely"
    )
    parser.add_argument('-i', action='store', nargs='+', default=[],
                        help='Path to two input json files')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
