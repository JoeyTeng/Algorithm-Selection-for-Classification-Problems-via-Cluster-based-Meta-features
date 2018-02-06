
# @Author: Joey Teng
# @Email:  joey.teng.dev@gmail.com
# @Filename: convert.py
# @Last modified by:   Joey Teng
# @Last modified time: 06-Feb-2018

import sys
import itertools
import json

import numpy

if __name__ == '__main__':
    if 'Start: Version' in open('{0}.log'.format(sys.argv[1])).read():
        raise ValueError("File Given is processed by a newer verson")
    clusters = json.load(open('{0}.clusters.json'.format(sys.argv[1]), 'r'))
    for cluster in itertools.chain(*clusters.values()):
        if (len(set([tuple(vertex) for vertex in cluster['vertices']]))
           < len(cluster['vertices'][0])):
            cluster['volume'] = numpy.exp(cluster['volume'])
    json.dump(
        clusters,
        open('{0}.clusters.v1.0.1.json'.format(sys.argv[1]), 'w'))
