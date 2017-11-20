# @Author: Joey Teng <Toujour>
# @Date:   20-Nov-2017
# @Email:  joey.teng.dev@gmail.com
# @Filename: convex_hull_cluster.py
# @Last modified by:   Toujour
# @Last modified time: 20-Nov-2017
"""
Input argument list: dataset_filename, clusters_filename, output_filename, log_file(optional)
Predefined types
    Point: <'list'>
        {'coordinate': [float, ...], 'label': int}
"""

import json
import logging


def initialize_logger(log_file=None, level=logging.INFO, filemode='w'):
    format = '%(asctime)s %(levelname)s\n  %(filename)s:%(lineno)s: %(name)s %(message)s'

    if log_file is None:
        handler = logging.StreamHandler()
    else:
        handler = logging.handlers.FileHandler(
            filename=log_file, filemode=filemode)

    handler.setFormatter(logging.Formatter(format))
    logger = logging.getLogger('LOG')
    logger.addHandler(handler)
    logger.setLevel(level)

    return logger


def load_dataset(filename):
    """
    argument:
        filename: path of input file.
            CSV format
            [coordinate, ...] + [label]

    return:
        dataset:
            list of dict objects:
            [Point, ...]
    """
    pass


def clustering(dataset):
    """
    Description:
        Convex Hull Algorithm - modified
        Base on Gift Wrapping
        All hulls will be pure (only contains data points with same label)

    argument:
        dataset:
            list of dict objects:
            [Point, ...]

    return:
        clusters:
            list of dict objects:
            [{'vertices': [Point, ...],
              'points': [Point, ...] (vertices are excluded)
              'size': int = len(['vertices']) + len(['points']),
              'volume': float (optional)}, ...]
    """
    pass


def size_versus_number_of_clusters(clusters):
    pass


def volume_versus_size(clusters):
    pass


if __name__ == '__main__':
    dataset_filename, clusters_filename, output_filename, log_file = sys.argv[1:] + [
        None]

    logger = initialize_logger(log_file)
    logger.info('Start')
    logger.debug('Logger initialized')
    logger.debug('sys.argv: {0}'.format(sys.argv))

    logger.debug('Loading dataset')
    dataset = load_dataset(dataset_filename)
    logger.info('Dataset loaded')

    logger.debug('Clustering data points')
    clusters = clustering(dataset)
    logger.debug(
        'Dumping clusters data into json file: {0}'.format(clusters_filename))
    json.dump(clusters, open(clusters_filename, 'w'))
    logger.info('Data points clustered')

    logger.debug('Calculating meta-feature indicators')
    features = {'Number of Clusters': len(clusters),
                'Size versus Number of Clusters': size_versus_number_of_clusters(clusters),
                'Volume versus Size': volume_versus_size(clusters)}
    logger.debug(
        'Dumping meta-feature indicators into json file: {0}'.format(clusters_filename))
    json.dump(features, open(output_filename, 'w'))
    logger.info('Meta-feature indicators calculated')

    logger.info('Completed')
