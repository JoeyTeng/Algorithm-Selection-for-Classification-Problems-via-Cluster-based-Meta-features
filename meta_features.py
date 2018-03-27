
# @Author: Joey Teng
# @Email:  joey.teng.dev@gmail.com
# @Filename: meta_features.py
# @Last modified by:   Joey Teng
# @Last modified time: 27-Mar-2018
"""Define and calculate meta-features using given clusters.

See function meta-features()
"""

import collections
import itertools
import math

import numpy


def size_versus_number_of_clusters(clusters):
    """Calculate the number of clusters respect to each size.

    Args:
        clusters (list): list of clusters

    Returns:
        dict:
            float: average
            float: standard deviation
            int: range
            dict: stats
                {size (int): quantity (int), ...}

    """
    stats = collections.defaultdict(int)  # default = 0
    sizes = [cluster['size'] for cluster in clusters]
    for cluster in clusters:
        # initial quantity is 0
        stats[cluster['size']] += 1

    average = numpy.average(sizes)
    standard_deviation = numpy.std(sizes)
    range_ = max(sizes) - min(sizes)

    return {
        'average': average,
        'standard deviation': standard_deviation,
        'range': range_,
        'stats': stats}


def volume_versus_size(clusters):
    """Calculate volume of clusters respect to its size.

    Args:
        clusters (list): list of clusters

    Returns:
        dict: {size (int): volume (list of floats)}

    """
    stats = collections.defaultdict(list)
    for cluster in clusters:
        # initial container is empty
        stats[cluster['size']].append(cluster['volume'])
    return stats


def calculate_density(cluster):
    """Calculate Density of a cluster.

    density = size / volume

    Args:
        clusters (list): list of clusters

    Returns:
        float: density

    """
    if numpy.isclose([cluster['volume']], [0]):
        return float('inf')

    density = cluster['size'] / cluster['volume']
    return density


def density_distribution(clusters, slots):
    """Calculate number of clusters in each density interval.

    [lb - 1 * interval, ... (slots - 1) * interval - hb]
    lb = lower bound
    hb = higher bound
    interval = range / slots = (hb - lb) / slots

    Args:
        clusters (list): list of clusters
        slots (int): number of intervals

    Returns:
        dict:
            float: interval
                range / slots
            float: average
                numpy.average
            float: standard deviation
                numpy.std
            float: range
                higherbound - lowerbound
            dict: stats
                from lower bound to higher
                {inf: int, n-th slot: int, ...}
                [lb - 1 * interval, ... (slots - 1) * interval - hb]

    """
    raw_densities = list(map(calculate_density, clusters))
    densities = [
        density for density in raw_densities
        if math.isfinite(density)]

    stats = collections.defaultdict(int)
    stats[float('inf')] = len(list(raw_densities)) - len(densities)
    interval = None
    lowerbound = float('inf')
    higherbound = float('-inf')
    if densities:
        lowerbound = min(densities)
        higherbound = max(densities)
        _range = higherbound - lowerbound
        interval = _range / slots
        if numpy.isclose([interval], [0]):
            interval = lowerbound

        for density in densities:
            stats[int((density - lowerbound) / interval)] += 1

    average = numpy.average(densities)
    standard_deviation = numpy.std(densities)
    range_ = higherbound - lowerbound

    return {'interval': interval,
            'min': lowerbound,
            'average': average,
            'standard deviation': standard_deviation,
            'range': range_,
            'stats': stats}


def label_versus_meta_features(clusters, func, *args, **kwargs):
    """Calculate meta-features for clusters with each label.

    Separate clusters based on label and call the funcitons
    Include a '_population' label which indicate the meta-feature over
        the population regardless of the label

    Args:
        clusters (dict): list of clusters with ['label']
        func (function):
            the function that used to calculate the meta-feature required

    Returns:
        dict: stats
            {label (label): corresponding meta-feature, ...}

    """
    _clusters = collections.defaultdict(list)
    _clusters['_population'] = list(itertools.chain(*clusters.values()))
    _clusters.update(clusters.items())
    stats = {}
    for label in _clusters:
        stats[label] = func(_clusters[label], *args, **kwargs)
    return stats


def meta_features(clusters):  # TODO
    """Calculate all the meta-features defined using clusters calculated.

    Args:
        clusters (list): list of clusters
            [{
                'vertices' (list): vertices
                    all the vertices on/defined the hull
                'points' (list): vertices
                    all the instances that are in the hull
                    (same label as homogeniety is maintained)
                'size' (int): the number of instances belong to this hull
                    len(vertices) + len(points)
                'volume' (float):
                    the volume in the Euclidean n-dimensional space obtained
                    by the hull
                'label' (int):
                    the category that the hull belongs to
            }, ...]

    Returns:
        meta-features (dict):
            {
                'Number of Clusters' (int)
                'Size versus Number of Clusters' ():
                'Volume versus Size' ():
                'Density distribution over 10 intervals' ():
            }

    """
    return {'Number of Clusters':
            label_versus_meta_features(clusters, len),
            'Size versus Number of Clusters':
                label_versus_meta_features(
                    clusters, size_versus_number_of_clusters),
            'Volume versus Size':
                label_versus_meta_features(clusters, volume_versus_size),
            'Density distribution over 10 intervals':
                label_versus_meta_features(clusters, density_distribution, 10)}
