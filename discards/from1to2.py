# @Author: Joey Teng <Toujour>
# @Date:   14-Nov-2017
# @Email:  joey.teng.dev@gmail.com
# @Filename: From1to2.py
# @Last modified by:   Toujour
# @Last modified time: 14-Nov-2017

import sys


def load(filename):
    import json

    return json.load(open(filename, 'r'))


def save(clusters, filename):
    import json

    json.dump(clusters, open(filename, 'w'))


def main(clusters, filename):
    import numpy

    for cluster in clusters:
        cluster['size'] = 0

    for line in open(filename, 'r'):
        position = list(map(float, line.split(',')))[:-1]

        for cluster in clusters:
            distance = numpy.linalg.norm(numpy.array(
                cluster['centroid']['dimension']) - numpy.array(position))
            if distance <= cluster['radius']:
                cluster['size'] += 1

    return clusters


if __name__ == '__main__':
    print("INFO: Start", flush=True)
    clusters = load(sys.argv[2])
    clusters = main(clusters, sys.argv[1])
    save(clusters, sys.argv[2] + '.json')
    print("INFO: Complete", flush=True)
