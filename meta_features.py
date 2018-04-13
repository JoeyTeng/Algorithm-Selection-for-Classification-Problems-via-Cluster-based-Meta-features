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


INFINITESIMAL = 1e-323


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


def log_volume_versus_size(clusters):
    """Calculate log-volume of clusters respect to its size.

    Args:
        clusters (list): list of clusters

    Returns:
        dict: {size (int): log-volume (list of floats)}

    """
    stats = collections.defaultdict(list)
    for cluster in clusters:
        # initial container is empty
        stats[cluster['size']].append(cluster['log-volume'])
    return stats


def calculate_inverse_density(cluster):
    """Calculate the inverse of Density of a cluster.

    inverse of density = volume / size

    Args:
        clusters (list): list of clusters

    Returns:
        float: inverse of density

    """
    inverse_density = cluster['volume'] / cluster['size']
    return inverse_density


def inverse_density_distribution(clusters, slots):
    """Calculate number of clusters in each inverse of density interval.

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
    inverse_densities = list(map(calculate_inverse_density, clusters))

    stats = collections.defaultdict(int)
    interval = None
    lowerbound = INFINITESIMAL
    higherbound = INFINITESIMAL
    if inverse_densities:
        lowerbound = min(inverse_densities)
        higherbound = max(inverse_densities)
        _range = higherbound - lowerbound
        interval = _range / slots
        if math.isclose(interval, 0):
            interval = max(lowerbound, float(1))  # prevent ZeroDivisionError

        for inverse_density in inverse_densities:
            try:
                stats[int((inverse_density - lowerbound) / interval)] += 1
            except ZeroDivisionError:
                print("Densities: {}".format(inverse_densities))
                print("Volumes: {}".format(
                    list(map(lambda x: x['volume'], clusters))))
                print("Size: {}".format(
                    list(map(lambda x: x['size'], clusters))))
                raise ZeroDivisionError(
                    "({} - {}) / {}".format(
                        inverse_density, lowerbound, interval))
            except ValueError as message:
                print("Densities: {}".format(inverse_densities))
                print("Volumes: {}".format(
                    list(map(lambda x: x['volume'], clusters))))
                print("Size: {}".format(
                    list(map(lambda x: x['size'], clusters))))
                raise ValueError(
                    "({} - {}) / {}\n{}".format(
                        inverse_density, lowerbound, interval, message))

    average = numpy.average(inverse_densities)
    standard_deviation = numpy.std(inverse_densities)
    range_ = higherbound - lowerbound

    return {'interval': interval,
            'min': lowerbound,
            'average': average,
            'standard deviation': standard_deviation,
            'range': range_,
            'stats': stats}


def calculate_inverse_log_density(cluster):
    """Calculate the log of inverse of Density of a cluster.

    inverse of density-log = log-volume - ln(size)

    Args:
        cluster ():

    Returns:
        float: inverse of density-log
               -inf if log-volume = -inf

    """
    inverse_log_density = cluster['log-volume'] - math.log(cluster['size'])
    return inverse_log_density


def inverse_log_density_distribution(clusters, slots):
    """Calculate number of clusters in each inverse of density interval.

    inverse_log_density = log-volume - ln(size)

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
    raw_inverse_log_densities = list(
        map(calculate_inverse_log_density, clusters))
    inverse_log_densities = [
        inverse_log_density
        for inverse_log_density in raw_inverse_log_densities
        if math.isfinite(inverse_log_density)]

    stats = collections.defaultdict(int)
    interval = None
    lowerbound = INFINITESIMAL
    higherbound = INFINITESIMAL
    if inverse_log_densities:
        lowerbound = min(inverse_log_densities)
        higherbound = max(inverse_log_densities)
        _range = higherbound - lowerbound
        interval = _range / slots
        if math.isclose(interval, 0):
            interval = max(lowerbound, float(1))  # prevent ZeroDivisionError

        for inverse_log_density in inverse_log_densities:
            try:
                stats[int((inverse_log_density - lowerbound) / interval)] += 1
            except ZeroDivisionError:
                print("Densities: {}".format(inverse_log_densities))
                print("Volumes: {}".format(
                    list(map(lambda x: x['volume'], clusters))))
                print("Size: {}".format(
                    list(map(lambda x: x['size'], clusters))))
                raise ZeroDivisionError(
                    "({} - {}) / {}".format(
                        inverse_log_density, lowerbound, interval))
            except ValueError as message:
                print("Densities: {}".format(inverse_log_densities))
                print("Volumes: {}".format(
                    list(map(lambda x: x['volume'], clusters))))
                print("Size: {}".format(
                    list(map(lambda x: x['size'], clusters))))
                raise ValueError(
                    "({} - {}) / {}\n{}".format(
                        inverse_log_density, lowerbound, interval, message))

        # All spheres with -inf volume
        stats[-1] = len(raw_inverse_log_densities) - len(inverse_log_densities)

    average = numpy.average(inverse_log_densities)
    standard_deviation = numpy.std(inverse_log_densities)
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
                'Inverse Density distribution over 10 intervals' ():
            }

    """
    return {'Number of Clusters':
            label_versus_meta_features(clusters, len),
            'Size versus Number of Clusters':
                label_versus_meta_features(
                    clusters, size_versus_number_of_clusters),
            # 'Volume versus Size':
            #     label_versus_meta_features(clusters, volume_versus_size),
                'log-Volume versus Size':
                label_versus_meta_features(clusters, log_volume_versus_size),
            # 'Inverse Density distribution over 10 intervals':
            #     label_versus_meta_features(
            #         clusters, inverse_density_distribution, 10)
            'Inverse Log Density distribution over 10 intervals':
                label_versus_meta_features(
                    clusters, inverse_log_density_distribution, 10)}
