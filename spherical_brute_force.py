# @Author: Joey Teng
# @Email:  joey.teng.dev@gmail.com
# @Filename: spherical_brute_force.py
# @Last modified by:   Joey Teng
# @Last modified time: 31-Jul-2018
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
import argparse
import collections
import json
import logging
import logging.handlers
import math
import os

import numpy

import meta_features


INFINITESIMAL = 1e-323
PROCESS_COUNT = int(os.cpu_count() / 2)


def initialize_logger(
        name='LOG',
        filename=None,
        level=logging.DEBUG,
        filemode='a'):
    """Initialize a logger in module logging.

    Args:
        name (string, optional): Name of logger. Defaults to None.
        filename (string, optional): Defaults to None.
            The path of log file
            By default, logger will stream to the standard output
        level (logging level, optional): Defaults to logging.INFO
        filemode (string, optional): Defaults to 'a'.
            'w' or 'a', overwrite or append

    Returns:
        logger: [description]

    """
    log_format = '%(asctime)s %(levelname)s\n' + \
        '  %(filename)s:%(lineno)s: %(name)s: %(message)s'

    if filename is None:
        handler = logging.StreamHandler()
    else:
        handler = logging.handlers.RotatingFileHandler(
            filename=filename, mode=filemode)

    handler.setFormatter(logging.Formatter(log_format))
    logger = logging.getLogger(name)
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
        'log-volume': calculate_log_volume(len(centroid), radius)
    }


def calculate_distance(lhs, rhs):
    """Calculate the euclidean distance between 2 points.

    Args:
        lhs, rhs (Vertex): Coordinates of 2 points

    Returns:
        float: Euclidean distance between them

    """
    return numpy.linalg.norm((numpy.array(lhs) - numpy.array(rhs)))


def calculate_log_volume(dimension, radius):
    """Calculate the log-volume of a sphere with given dimension and radius.

    Args:
        dimension (int): dimension of the space
        radius (float): radius of the sphere

    Returns:
        float: the log-volume of the sphere
               radius is set as REL_TOL (1e-09)

    """
    if (math.isclose(radius, 0)):
        radius = INFINITESIMAL

    try:
        log_volume = ((dimension / 2.0) * math.log(math.pi) + dimension *
                      math.log(radius) - math.lgamma(dimension / 2.0 + 1))
    except ValueError as message:
        raise ValueError("".join([
            "{0}\n".format(message),
            "(({0} / 2.0) * ln(pi) + ({0} * ln({1})".format(dimension, radius),
            " - ln(gamma({0} / 2.0 + 1)))".format(dimension)]))
    if math.isnan(log_volume):
        raise ValueError(
            "Volume is NaN: pi ^ " +
            "({0} / 2.0) / gamma({0} / 2.0 + 1) * {1} ^ {0}".format(
                dimension, radius))

    return log_volume


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


def check_inside_cluster(cluster, point):
    """Check if point is inside the cluster.

    Args:
        cluster (dict): cluster to be checked
            {
                'centroid' (Vertex): centroid of the cluster,
                'radius' (float): radius of the cluster
            }
        point (Vertex): point to be checked

    Returns:
        bool: if the point is encompassed by the boundary

    """
    return float_less_or_equal(
        calculate_distance(cluster['centroid'], point), cluster['radius'])


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


def check_homogeneity_instances(indices, dataset):
    labels = set(map(lambda x: dataset[x]['label'], indices))
    if len(labels) > 1:
        return False
    return True


def sub_partitions(indices, n, current):
    # n (int) is the number of groups
    # current (list) is the current grouping
    r = len(indices)
    # print(indices, n, current)
    if n == 1:
        yield [list(indices)]
        return
    if n == r:
        for i, index in enumerate(indices):
            tmp = [current + [index]]
            tmp.extend(list(map(lambda x: [x], indices[:i] + indices[i + 1:])))
            yield tmp
        return

    for other in sub_partitions(indices[1:], n - 1, []):
        tmp = [current + [indices[0]]]
        tmp.extend(other)
        yield tmp

    for index in range(1, len(indices)):
        indices[1], indices[index] = indices[index], indices[1]
        for tmp in sub_partitions(indices[1:], n, current + [indices[0]]):
            yield tmp
        indices[1], indices[index] = indices[index], indices[1]

    return


def partition(indices):
    r = len(indices)
    for n in range(1, r + 1):
        for tmp in sub_partitions(indices[:], n, []):
            yield tmp


def clustering(dataset, logger):
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

    clusters = collections.defaultdict(list)
    instances = [instance['coordinate'] for instance in dataset]

    count = 0
    found_count = 0
    minimum = len(instances)
    logger.info('Checking clusters...')
    for groups in partition(list(range(len(dataset)))):
        tmp_clusters = collections.defaultdict(list)

        if len(groups) > minimum:
            logger.info('Minimum found. #groups: {}'.format(len(groups)))
            break

        for indices in groups:
            cluster = initialize_cluster(list(
                map(lambda x: instances[x], indices)))
            label = dataset[indices[0]]['label']
            if (not check_homogeneity(cluster, label, tmp_clusters)
                    or not check_homogeneity_instances(indices, dataset)):
                break
            tmp_clusters[label].append(cluster)
        else:
            minimum = len(groups)
            clusters = tmp_clusters
            logger.info('Minimum updated. #{} group'.format(count))

            found_count += 1
            logger.info(
                'One option found. Total till now: {}'.format(found_count))
        count += 1
        if count % 50 == 0:
            logger.info('{} groupings checked'.format(count))

    return clusters


def main(args):
    """
    Start main function here.

    Dispatching all the tasks to process.
    """
    log_file = args.log

    logger, handler = initialize_logger("Parent", log_file)
    logger.info('Start: Version 2.1.1')
    logger.debug('Logger initialized')
    logger.debug('argparse: %r', args)
    logger.removeHandler(handler)

    _args = []
    for dataset_filename in args.paths:
        clusters_filename = dataset_filename + ".clusters.json"
        output_filename = dataset_filename + ".output.json"

        _args.append(tuple([
            dataset_filename,
            clusters_filename,
            output_filename,
            log_file]))

    list(map(task_processing, _args))


def task_processing(args):  # Take note here!!!
    """Unwrap the args tuple to adapt a function with multiple args to map."""
    def worker(
            dataset_filename,
            clusters_filename,
            output_filename,
            log_file):
        """Link the submodules to process the data."""
        logger, handler = initialize_logger(dataset_filename, log_file)
        logger.debug('Logger initialized')

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

        logger.info('Complete')

        logger.removeHandler(handler)

    return worker(*args)


def traverse(paths):
    """Traverse to collect all the data files."""
    print("Starting Traverse Through", flush=True)
    files = []
    while paths:
        path = paths[0]
        paths = paths[1:]
        for file in os.listdir(path):
            if (file.find('.json') == -1
                    and file.find('.log') == -1
                    and file.find('.DS_Store') == -1
                    and file.find('.png') == -1
                    and file.find('.html') == -1):
                files.append('{0}/{1}'.format(path, file))
            elif os.path.isdir('{0}/{1}'.format(path, file)):
                paths.append('{0}/{1}'.format(path, file))

    print("Traverse Completed.", flush=True)
    return files


def parse_args():
    """Parse all necessary args."""
    parser = argparse.ArgumentParser(
        description="Obtain clusters and calculate meta-features")
    parser.add_argument('-r', action='store', nargs='+',
                        default=[], metavar='Directory',
                        help='Recursively processing all files in the folder')
    parser.add_argument('-i', action='store', nargs='+',
                        default=[], metavar='File',
                        help='Files that need to be processed')
    parser.add_argument('--log', action='store', type=str,
                        default='spherical_cluster.log', metavar='Log file',
                        help='Path to the log file')

    args = parser.parse_args()
    paths = []
    if (args.r):
        paths = traverse(args.r)
    paths.extend(args.i)
    paths.sort()
    args.paths = paths

    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)
