# @Author: Joey Teng
# @Email:  joey.teng.dev@gmail.com
# @Filename: spherical_cluster.py
# @Last modified by:   Joey Teng
# @Last modified time: 25-Mar-2018
"""Obtain clusters and calculate meta-features.

Args:
    dataset_filename (string): path to the dataset

Predefined types:
    Point (dict): {'coordinate': (float, ...), 'label': int}
    Dataset (list): list of dict objects:
        [Point, ...]
    Vertex (tuple): Point['coordinate']
    Vertices (list): [Vertex, ...]

Output files:
    dataset_filename.output.json: calculated meta-features.
    dataset_filename.clusters.json: calculated clusters.
    dataset_filename.log: log file

"""
import collections
import json
import logging
import logging.handlers
import sys

import numpy
import scipy.special

import meta_features


def initialize_logger(filename=None, level=logging.DEBUG, filemode='w'):
    """Initialize a logger in module logging.

    Args:
        filename (string, optional): Defaults to None.
            The path of log file
            By default, logger will stream to the standard output
        level (logging level, optional): Defaults to logging.INFO
        filemode (string, optional): Defaults to 'w'.
            'w' or 'a', overwrite or append

    Returns:
        logger: [description]

    """
    log_format = '%(asctime)s %(levelname)s\n' + \
        '  %(filename)s:%(lineno)s: %(name)s %(message)s'

    if filename is None:
        handler = logging.StreamHandler()
    else:
        handler = logging.handlers.RotatingFileHandler(
            filename=filename, mode=filemode)

    handler.setFormatter(logging.Formatter(log_format))
    logger = logging.getLogger('LOG')
    logger.addHandler(handler)
    logger.setLevel(level)

    return logger, handler


def load_dataset(filename):
    """Load data from a csv file.

    Args:
        filename (string): path of input file.
            CSV format
            [coordinate, ...] + [label]

    Returns:
        Dataset: dataset

    """
    return [(
        lambda point: {
            'coordinate': tuple(map(float, point[:-1])),
            'label': int(point[-1])})
            (string.strip().rstrip().split(','))
            for string in open(filename, 'r').read()
            .strip().rstrip().split('\n')]


def initialize_cluster(coordinates):
    """Construct a cluster instance with given coordiante.

    A factory function

    Args:
        coordinates (list): The coordinates that needed to be included.
            [Vertex, ...]

    Returns:
        dict: a cluster initialized with given coordinates
            [{
                'centroid' (Vertex): centroid of the sphere,
                'radius' (float): radius of the sphere,
                'points' (list): Instances in the cluster
                        i.e. distance <= radius
                    [Vertex, ...],
                'size' (int): Number of instances covered by the sphere
                    len(['points']),
                'volume' (float): volume of the sphere
            }]

    """
    points = coordinates
    _points = list(map(numpy.array, coordinates))
    centroid = sum(_points) / len(_points)
    radius = max(
        map(lambda x, y=centroid: numpy.linalg.norm((x - y)), _points))

    return {
        'centroid': tuple(centroid),
        'radius': radius,
        'points': points,
        'size': len(points),
        'volume': calculate_volume(len(centroid), radius)
    }


def calculate_distance(lhs, rhs):
    """Calculate the euclidean distance between 2 points.

    Args:
        lhs, rhs (Vertex): Coordinates of 2 points

    Returns:
        float: Euclidean distance between them

    """
    return numpy.linalg.norm((numpy.array(lhs) - numpy.array(rhs)))


def calculate_volume(dimension, radius):
    """Calculate the volume of a sphere with given dimension and radius.

    Args:
        dimension (int): dimension of the space
        radius (float): radius of the sphere

    Returns:
        float: the volume of the sphere

    """
    return ((numpy.pi ** (dimension / 2.0))
            / scipy.special.gamma(dimension/2.0 + 1)
            * radius ** dimension)


def float_less_or_equal(lhs, rhs, **kwargs):
    """Determine float A is less than or equal to B using numpy.isclose().

    Use numpy.isclose() to determine if A and B are equal
        with default tolerance.

    Args:
        lhs, rhs (float): values that need to be compared
        kwargs: kwargs for numpy.isclose()

    Returns:
        bool: result of comparison.

    """
    return numpy.isclose(lhs, rhs, **kwargs) or (lhs < rhs)


def check_homogeneity(cluster, label, clusters):
    """Check homogeneity of the cluster with given clusters.

    A homogeneous cluster will not overlap with any other cluster which has
        different label, but may overlap with cluster that has the same label.
        Which means, there should be no region with ambiguity in
        categorisation process.

    Args:
        cluster (dict): Cluster that need to be checked
            {
                'centroid' (Vertex): centroid of the cluster,
                'radius' (float): radius of the cluster
            }
        label (): label of the cluster
        clusters (dict): list of clusters with labels as keys.
            {
                label: [cluster, ...]
            }

    Returns:
        bool: if cluster is homogeneous

    """
    for _label, _clusters in clusters.items():
        if _label == label:
            continue

        for _cluster in _clusters:
            if float_less_or_equal(
                calculate_distance(
                    cluster['centroid'], _cluster['centroid']),
                    (cluster['radius'] + _cluster['radius'])):
                return False
    return True


def clustering(dataset, logger):  # TODO
    """Calculate all spherical clusters.

    All spheres will be pure(only contains data points with same label)

    Args:
        dataset (list): All the instances in the space with label
            list of dict objects:
            [Point, ...]
        logger (logger): logger for logging

    Returns:
        dict: Clusters obtained separated by labels
            label: clusters (list of dict objects)
                [{
                'centroid' (Vertex): centroid of the sphere,
                'radius' (float): radius of the sphere,
                'points' (list) : Instances in the cluster
                    [Vertex, ...],
                'size' (int): Number of instances covered by the sphere
                    len(['points']),
                'volume': The volume of the sphere
                    float(optional)
                }, ...]

    """
    logger.info('Sorting datasets...')
    dataset.sort(key=lambda x: x['coordinate'])

    logger.info('Initialise clusters...')
    clusters = collections.defaultdict(list)
    for instance in dataset:
        clusters[instance['label']].append(
            initialize_cluster((instance['coordinate'], )))

    logger.info('Merging clusters...')
    logger_count = 0
    for label, homo_clusters in clusters.items():
        index = 0
        while index < len(homo_clusters):
            current = homo_clusters[index]
            merging_index = -1
            distance = float('inf')
            for j_index, cluster in enumerate(homo_clusters[index + 1:]):
                new_distance = calculate_distance(
                    current['centroid'], cluster['centroid'])
                if new_distance < distance:
                    merging_index = j_index + index + 1
                    distance = new_distance

            if merging_index == -1:
                index += 1
                continue

            cluster = initialize_cluster(
                current['points'] + homo_clusters[merging_index]['points'])
            if (check_homogeneity(cluster, label, clusters)):
                homo_clusters[merging_index], homo_clusters[-1] =\
                    homo_clusters[-1], homo_clusters[merging_index]
                homo_clusters.pop()
                current = cluster
                homo_clusters[index] = current
            else:
                index += 1
        logger_count += 1
        logger.info('{0}/{1} categories completed'.format(
            logger_count, len(clusters.keys())))

    return clusters


def main(argv):
    """Start main function here."""
    dataset_filename = argv[0]
    clusters_filename = dataset_filename + ".clusters.json"
    output_filename = dataset_filename + ".output.json"
    log_file = dataset_filename + ".log"

    logger, handler = initialize_logger(log_file)
    logger.info('Start: Version 0.0.1')
    logger.debug('Logger initialized')
    logger.debug('sys.argv: %r', sys.argv)

    logger.debug('Loading dataset')
    dataset = load_dataset(dataset_filename)
    logger.info('Dataset loaded')

    logger.info('Trying to load clusters from %s', clusters_filename)
    clusters = None
    try:
        clusters = json.load(open(clusters_filename, 'r'))
    except FileNotFoundError:
        logger.warning('Clusters data file not found')
    except json.decoder.JSONDecodeError:
        logger.warning('File broken. Not Json Decodable')

    if not clusters:
        logger.debug('Clustering data points')
        clusters = clustering(dataset, logger)
        logger.debug(
            'Dumping clusters data into json file: %s', clusters_filename)
        json.dump(clusters, open(clusters_filename, 'w'))
        logger.info('Data points clustered')

    logger.debug('Calculating meta-feature indicators')
    features = meta_features.meta_features(clusters)
    logger.debug(
        'Dumping meta-feature indicators into json file: %s',
        clusters_filename)
    json.dump(features, open(output_filename, 'w'))
    logger.info('Meta-feature indicators calculated')

    logger.info('Completed')
    logger.removeHandler(handler)


if __name__ == '__main__':
    main(sys.argv[1:])
