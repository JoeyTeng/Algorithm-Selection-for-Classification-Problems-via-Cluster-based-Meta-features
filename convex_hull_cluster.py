# @Author: Joey Teng <Toujour>
# @Date:   20-Nov-2017
# @Email:  joey.teng.dev@gmail.com
# @Filename: convex_hull_cluster.py
# @Last modified by:   Toujour
# @Last modified time: 24-Jan-2018
"""
Input argument list:
    dataset_filename
Predefined types:
    Point: <'dict'>
        {'coordinate': (float, ...), 'label': int}

    Dataset: <'list'>
        list of dict objects:
        [Point, ...]

    Vertex: <'tuple'>
        Point['coordinate']

    Vertices: <'list'>
        [Vertex, ...]
"""
import collections
import itertools
import json
import logging
import logging.handlers
import math
import queue
import sys

import numpy


def initialize_logger(filename=None, level=logging.INFO, filemode='w'):
    """Initialize a logger in module logging

    Args:
        filename:
            <type>: <string>
                the path of log file
            <default>: None
                stream to the standard output
        level:
            logging level
        filemode:
            'w' or 'a', overwrite or append

    Returns:
        logger
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

    return logger


def load_dataset(filename):
    """Load data from a csv file

    Args:
        filename: path of input file.
            CSV format
            [coordinate, ...] + [label]

    Returns:
        dataset:
            <type>: Dataset
    """
    return [(
        lambda point: {
            'coordinate': tuple(map(float, point[:-1])),
            'label': int(point[-1])})
            (string.strip().rstrip().split(','))
            for string in open(filename, 'r').read()
            .strip().rstrip().split('\n')]


def signed_volume(vertices):
    """
    Description:
        Calculate the signed volume of n-dimensional simplex
            defined by (n + 1) vertices
        Reference:
            Wedge Product: http://mathworld.wolfram.com/WedgeProduct.html

    Parameters:
        vertices:
            <type>: Vertices

    Returns:
        sign : (...) array_like
            A number representing the sign of the determinant. For a real
                matrix, this is 1, 0, or -1. For a complex matrix, this is a
                complex number with absolute value 1 (i.e., it is on the unit
                circle), or else 0.
        logdet : (...) array_like
            The natural log of the absolute value of the determinant.

        If the determinant is zero, then sign will be 0 and logdet will be
        -Inf. In all cases, the determinant is equal to sign * np.exp(logdet).

        Reference:
            From scipy manual
                https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.linalg.slogdet.html#numpy.linalg.slogdet
    """
    dimension = len(vertices[0])
    (sign, logvolume) = numpy.linalg.slogdet(
        numpy.stack(vertices[1:]) +
        numpy.array(vertices[0]) * numpy.ones((dimension, dimension)) * -1)
    return (sign, logvolume)


def squared_area(vertices):
    """
    Description:
        Calculate the squared area of (n - 1)-dimensional simplex defined by
            n vertices in n-dimensional space
        Reference:
            Wedge Product: http://mathworld.wolfram.com/WedgeProduct.html

    Parameters:
        vertices:
            <type>: Vertices

    Returns:
        logdet : (...) array_like
            The natural log of the absolute value of the determinant.

        If the determinant is zero, then sign will be 0 and logdet will be
        -Inf. In all cases, the determinant is equal to sign * np.exp(logdet).

        Reference:
            From scipy manual
                https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.linalg.slogdet.html#numpy.linalg.slogdet
    """
    dimension = len(vertices[0])
    matrix = numpy.matrix(
        numpy.stack(vertices[1:]) +
        numpy.array(vertices[0]) * numpy.ones((dimension - 1, dimension)) * -1)
    logvolume = numpy.linalg.slogdet(matrix * matrix.T)[1]  # sign, logvolume
    return logvolume


def check_inside(face=None, pivot=None, edge=None, area=None):
    """
    Description
    Parameters:
        face:
        pivot:
            <type>: Vertex
        edge:
            Default:
                face[:-1]
        area:
            Default:
                squared_area(face)
    Returns:
        inside:
            <type>: bool
        face:
            new face generated with (edge + pivot)
        area:
            new squared_area calculated using _face
    """
    if face is None or pivot is None:
        raise ValueError(
            "Wrong parameters given: face is {0}, pivot is {1}".format(
                type(face), type(pivot)))
    edge = edge or face[:-1]
    area = area or squared_area(face)

    sign, logvolume = signed_volume(form_face(face, pivot))
    _face = form_face(edge, pivot)
    _area = squared_area(_face)
    if (numpy.isclose([numpy.e**logvolume], [0]) and _area > area) or sign < 0:
        # outside
        return (False, _face, _area)
    return (True, _face, _area)


def check_inside_hull(hull, pivot):
    """
    Description:
    Parameters:
    Returns:
        inside:
            <type>: bool
    """
    for face in hull:
        if not check_inside(face=face, pivot=pivot)[0]:
            return False
    return True


def check_homogeneity(all_instances, hull, label, used_pivots):
    """
    Description:
    Parameters:
        all_instances:
        hull:
        label:
        used_pivots:
            <type>: dict
                {Vertex: True}
    Returns:
        homogeneity:
            <type>: bool
    """
    for point in all_instances:
        pivot = point['coordinate']
        _label = point['label']
        if pivot in used_pivots or _label == label:
            continue
        if check_inside_hull(hull, pivot):
            return False

    return True


def pivot_on_edge(dataset, edge, label, used_pivots):
    """
    Description:
    Parameters:
        dataset:
            <type>: Dataset

        edge:
            <type>: Vertices

        label:
            <type>: Point['label']
    Recieve:
    Yields:
    Returns:
    """
    vertices_in_edge = set(edge)
    index = 0
    length = len(dataset)
    while index < length and \
            (dataset[index]['label'] != label or
             dataset[index]['coordinate'] in used_pivots):
        index += 1

    if index == length:
        yield (None, False)  # Not found
        return

    homo = {}
    homo['pivot'] = dataset[index]['coordinate']
    homo['face'] = form_face(edge, homo['pivot'])
    homo['area'] = squared_area(homo['face'])

    found = False
    check = yield [homo['pivot'], label, None]
    if check:
        found = True

    for point in dataset[index + 1:]:
        if point['label'] != label or point['coordinate'] in vertices_in_edge:
            # Skip all used pivots in edge to prevent self-orientating
            # Skip all instances labelled differently
            # Homogeneity test is checked every round
            continue

        current = {}
        current['pivot'] = point['coordinate']
        inside, current['face'], current['area'] = check_inside(
            face=homo['face'], pivot=current['pivot'],
            edge=edge, area=homo['area'])

        if not inside:
            check = yield [current['pivot'], label, None]
            if check:
                # update
                homo = current
                found = True

    yield (homo['pivot'], found)
    return


def find_next_pivot(dataset, hull, edge, label,
                    used_pivots, edge_count, all_instances):
    """
    Description:
    Parameters:
        dataset:
        hull:
        edge:
        label:

        used_pivots:
        edge_count:
        all_instances:
    Returns:
        pivot:
            Vertex
        found:
            bool
    """
    find_pivot = pivot_on_edge(dataset, edge, label, used_pivots)
    pivot = next(find_pivot)
    while len(pivot) == 3:
        # Find next pivot
        # Feedback: if the pivot suggested is a valid choice
        if pivot[0] in used_pivots:
            # Choose back will always generate a homogeneous hull
            # Skip the checking process
            pivot = find_pivot.send(True)
            continue

        check = {}
        check['_face'] = form_face(edge, pivot[0])
        hull.append(check['_face'])
        # Update Edge Count based on new face formed
        check['_edges'] = list(itertools.combinations(
            check['_face'], len(check['_face']) - 1))
        for _edge in check['_edges']:
            edge_count[_edge] += 1

        check['number of face added'] = close_up_hull(
            hull, edge_count, used_pivots)

        check['homogeneity'] = check_homogeneity(
            all_instances, hull, label, used_pivots)
        # Revert update
        while check['number of face added']:
            hull.pop()  # close_up
            check['number of face added'] -= 1

        for _edge in check['_edges']:
            edge_count[_edge] -= 1

        hull.pop()  # _face
        if check['homogeneity']:
            pivot = find_pivot.send(True)
        else:
            pivot = find_pivot.send(False)

    pivot, found = pivot
    if not found or pivot in used_pivots:
        # best next choice is used
        # stop searching and start closing up
        return (pivot, False)
    return (pivot, True)


def form_face(edge, pivot):
    """
    Description
    Parameters:
        edge:
            <type>: Vertices
        pivot:
            <type>: Point['coordinate']
    """
    return tuple(list(edge) + [pivot])


def close_up(edge_count, used_pivots):
    """
    Description:
    Parameters:
        edge_count:
        used_pivots:
    Returns:
        face:
    """
    edges = []
    for edge, count in edge_count.items():
        if count == 1:
            edges.append(edge)

    faces = []
    lazy_update = collections.defaultdict(int)  # default = 0
    while edges:
        vertices = None
        for (i, edge_a), (j, edge_b) in\
                itertools.combinations(enumerate(edges), 2):
            vertices = set(edge_a).union(set(edge_b))
            if len(vertices) == len(edge_a[0]):
                edges[i], edges[j], edges[-1], edges[-2] =\
                    edges[-1], edges[-2], edges[i], edges[j]
                edges.pop()
                edges.pop()
                break
        else:
            # Cannot find a face, update edges and edges count
            for edge in lazy_update:  # = .keys()
                if lazy_update[edge] + edge_count[edge] == 1:
                    edges.append(edge)
                    lazy_update[edge] = 2  # Avoid duplicated edges
            continue

        face = list(vertices)
        for pivot in used_pivots:  # = .keys()
            if pivot not in vertices:
                if not check_inside(face=face, pivot=pivot)[0]:
                    # det(A) = -det (B) if two cols swap (odd and even)
                    face[-1], face[-2] = face[-2], face[-1]
                break
        else:
            # This edge is the first edge
            return []

        faces.append(tuple(face))
        for edge in itertools.combinations(tuple(face), len(face) - 1):
            lazy_update[edge] += 1

    return faces


def close_up_hull(hull, edge_count, used_pivots):
    """
    Description:
        Second stage
        add all remaining faces into the hull to form
            a closed simplicial complex
    Parameters:
        hull:
        edge_count:
        used_pivots:
    Return:
        no_face_added:
            <type>: int
            Number of face added
    """
    face_added = close_up(edge_count, used_pivots)
    if not face_added:
        face = list(hull[0])
        # det(A) = -det (B) if two cols swap (odd and even)
        face[-2], face[-1] = face[-1], face[-2]
        face_added = [tuple(face)]
    for face in face_added:
        hull.append(face)

    return len(face_added)


def sort_vertices(*args, **kwargs):
    """
    A wrapper of sorting functions
    Using buitin sorted for now

    Args:
        same as the wrapped function

    Returns
        same as the wrapped function

    Raises:
        same as the wrapped fucntion
    """
    return sorted(*args, **kwargs)


def qsort_partition(data, target=1, lhs=0, rhs=None):
    """
    Description:
        Find the smallest [target] values in the [data] using [comp] as __lt__

    Complexity:
        O(n)

    Parameters:
        data:
            <type>: Dataset

        target:
            int
            [terget] smallest values will be returned.
            (default=1, smallest value will be returned in a list
        lhs:
            lowest index (default=0)
        rhs:
            highest index + 1 (default=len(data))
        comp: # Currently unavailable
            cumstomised function used for comparing
            default=__builtin__.__lt__

    Returns:
        list which contains [target] shallow copies of Vertex
    """
    lhs = lhs or 0
    rhs = (rhs or len(data)) - 1
    # comp is Partially supported: only used in partitioning
    # but not in sorting return values
    # BUG: Work around instead for now
    # comp = (lambda x, y: x < y)

    _data = set()
    label = data[0]['label']
    for element in data[lhs:rhs + 1]:
        if element['label'] == label:
            _data.add(element['coordinate'])
    data = list(_data)

    # BUG: Work around instead for now
    """
    rhs = len(data) - 1  # Since [data] is updated
    position = -1

    while position != target:
        if position < target:
            lhs = position + 1
        elif position > target:
            rhs = position - 1

        pivot = data[rhs]
        index = lhs
        for i in range(lhs, rhs + 1):
            if comp(data[i], pivot):
                data[i], data[index] = data[index], data[i]
                index += 1
        data[rhs], data[index] = data[index], data[rhs]
        position = index  # Return value
    return (label, sort_vertices(data[:target]))
    """
    return (label, sort_vertices(data)[:target])


def initialize_hull(dataset, all_instances):
    """
    Description
    Parameters:
        dataset:
            <type>: Vertices
    Returns:
        label:
        dimension:
            int
        face:
            (Vertex, ...)
        used_pivots:
    """
    dimension = len(dataset[0]['coordinate'])
    label, edge = qsort_partition(dataset, target=dimension - 1)
    used_pivots = dict(zip(edge, [True] * len(edge)))
    edge_count = collections.defaultdict(int)  # default = 0
    face = edge
    if len(edge) == dimension - 1:
        pivot, found = find_next_pivot(
            dataset, [], edge, label, used_pivots, edge_count, all_instances)
        if found:
            face = form_face(edge, pivot)
            used_pivots[pivot] = True
    return (label, dimension, tuple(face), used_pivots, edge_count)


def queuing_face(face, _queue, edge_count):
    """
    Description
    Parameters:
        face:
            <type>: Vertices
        _queue:
        edge_count:
            edge_count:
            <type>: dict
                {edge: int}
            Counting of the number of an edge is used
    """
    for i in range(len(face) - 1, -1, -1):
        sub_face = []
        for j, element in enumerate(face):
            if i != j:
                sub_face.append(element)
        edge = tuple(sub_face)
        sorted_edge = tuple(sort_vertices(edge))
        if not edge_count[sorted_edge]:
            _queue.put(edge)
        edge_count[sorted_edge] += 1


def gift_wrapping(dataset, all_instances):
    """
    Description:
    Parameters:
        dataset:
    Returns:
        <type>: dict
            {
                "faces": all the faces,
                    <type>: list
                    [face]
                "vertices": all the vertices
                    <type>: dict
                    {Vertex: True}
                "dimension":
                    <type>: int
                    len(face)
                "label":
                    <type>: int
                    label
            }
    Reference: https://www.cs.jhu.edu/~misha/Spring16/09.pdf
    """
    label, dimension, face, used_pivots, edge_count = initialize_hull(
        dataset, all_instances)
    _queue = queue.LifoQueue()
    if len(face) == dimension:
        queuing_face(face, _queue, edge_count)

    hull = []
    hull.append(face)
    vertices = [coordinate for coordinate in face]

    # First stage: find all new pivots
    while not _queue.empty():
        edge = _queue.get()
        pivot, found = find_next_pivot(
            dataset, hull, edge, label,
            used_pivots, edge_count, all_instances)
        if not found:
            continue

        face = form_face(edge, pivot)
        vertices.append(pivot)
        used_pivots[pivot] = True
        hull.append(face)
        queuing_face(face, _queue, edge_count)

    # Second stage: close up the hull
    if dimension < len(used_pivots):
        close_up_hull(hull, edge_count, used_pivots)
    return {
        "faces": hull,
        "vertices": used_pivots,
        "dimension": dimension,
        "label": label}


def clustering(dataset, logger):
    """
    Description:
        Convex Hull Algorithm - modified
        Base on Gift Wrapping
        All hulls will be pure(only contains data points with same label)

    Parameters:
        dataset:
            list of dict objects:
            [Point, ...]
        logger:

    Returns:
        clusters:
            list of dict objects:
            [{'vertices': [Vertex, ...],
              'points': [Vertex, ...](vertices are excluded)
              'size':
                    <type>: int
                    len(['vertices']) + len(['points']),
              'volume': float(optional)}, ...]
    """
    clusters = []
    all_instances = dataset
    while dataset:
        # List is not empty
        cluster = gift_wrapping(dataset, all_instances)

        found = cluster['dimension'] < len(cluster['vertices'])
        _dataset = []
        vertices = []
        points = []
        for point in dataset:
            if point['label'] != cluster['label']:
                _dataset.append(point)
                continue

            vertex = point['coordinate']
            if vertex in cluster['vertices']:
                vertices.append(vertex)
            else:
                if found and check_inside_hull(cluster['faces'], vertex):
                    points.append(vertex)
                else:
                    _dataset.append(point)

        if found:
            volume = calculate_volume(cluster['faces'])
        else:
            volume = 0

        dataset = _dataset
        clusters.append({'vertices': vertices,
                         'points': points,
                         'size': len(vertices) + len(points),
                         'volume': volume,
                         'label': cluster['label']})
        if len(clusters) % 100 == 0:
            logger.info(
                'Clustering: %d clusters found, %d/%d instance processed',
                len(clusters), len(all_instances) - len(dataset),
                len(all_instances))

    return clusters


def calculate_volume(hull):
    """
    Description:
    Parameter:
        hull:
    Return:
    """
    origin = hull[0][0]
    volume = 0.0
    for face in hull:
        logvolume = signed_volume(form_face(face, origin))[1]
        volume += numpy.e ** logvolume
    volume /= 2  # Triangles = det / 2

    return volume


def size_versus_number_of_clusters(clusters):
    """
    Description:
    Parameter:
        clusters
    Return:
        <type>: dict
            {
                size: quantity
                    int: int
            }
    """
    stats = collections.defaultdict(int)  # default = 0
    for cluster in clusters:
        # initial quantity is 0
        stats[cluster['size']] += 1
    return stats


def volume_versus_size(clusters):
    """
    Description:
    Parameter:
        clusters
    Return:
        <type>: dict
            {
                size: volume
                    int: [float]
            }
    """
    stats = collections.defaultdict(list)
    for cluster in clusters:
        # initial container is empty
        stats[cluster['size']].append(cluster['volume'])
    return stats


def centroid(clusters):
    """The centroid of the vertices on the convex hulls
        (i.e. exclude the inner instances)

    Args:
        clusters

    Returns:
        [vertex, ...]
    """
    centroids = list(map(
        lambda cluster: tuple(map(
            lambda x, cluster=cluster: x / len(cluster['vertices']),
            sum(map(
                numpy.array,
                cluster['vertices'])))),
        clusters))
    return centroids


def calculate_density(cluster):
    """Density of a cluster

    density = size / volume

    Args:
        cluster

    Returns:
        density:
            float
    """
    try:
        density = cluster['size'] / cluster['volume']
    except ZeroDivisionError:
        return float('inf')
    return density


def density_distribution(clusters, slots):
    """Number of clusters in each density interval

    [lb - 1 * interval, ... (slots - 1) * interval - hb]
    lb = lower bound
    hb = higher bound
    interval = range / slots = (hb - lb) / slots

    Args:
        slots:
            number of intervals

    Returns:
        interval:
            <type>: <float>
            range / slots
        stats:
            <type>: <dict>
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
    if densities:
        lowerbound = min(densities)
        higherbound = max(densities)
        _range = higherbound - lowerbound
        interval = _range / slots
        if numpy.isclose([interval], [0]):
            interval = lowerbound

        for density in densities:
            stats[int((density - lowerbound) / interval)] += 1

    return {'interval': interval,
            'stats': stats}


def label_versus_meta_features(clusters, func, *args, **kwargs):
    """Calculate meta-features for clusters with each label

    Separate clusters based on label and call the funcitons
    Include a '_population' label which indicate the meta-feature over
        the population regardless of the label

    Args:
        clusters:
        func:
            the function that used to calculate the meta-feature required

    Returns:
        stats:
            <type>: <dict>
            {
                label: corresponding meta-feature
            }
    """
    _clusters = collections.defaultdict(list)
    _clusters['_population'] = clusters
    for cluster in clusters:
        _clusters[cluster['label']].append(cluster)
    stats = {}
    for label in _clusters:
        stats[label] = func(_clusters[label], *args, **kwargs)
    return stats


def meta_features(clusters):
    """Calculating all the meta-features defined using clusters calculated.

    Args:
        clusters:
            <type>: <list>
            [{
                'vertices': vertices
                    <type>: list
                    all the vertices on/defined the hull
                'points': vertices
                    <type>: list
                    all the instances that are in the hull
                    (same label as homogeniety is maintained)
                'size': the number of instances belong to this hull
                    <type>: int
                    len(vertices) + len(points)
                'volume':
                    <type>: float
                    the volume in the Euclidean n-dimensional space obtained
                    by the hull
                'label':
                    <type>: int
                    the category that the hull belongs to
            }, ...]

    Returns:

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


def main(argv):
    """main"""
    dataset_filename = argv[0]
    clusters_filename = dataset_filename + ".clusters.json"
    output_filename = dataset_filename + ".output.json"
    log_file = dataset_filename + ".log"

    logger = initialize_logger(log_file)
    logger.info('Start')
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
    features = meta_features(clusters)
    logger.debug(
        'Dumping meta-feature indicators into json file: %s',
        clusters_filename)
    json.dump(features, open(output_filename, 'w'))
    logger.info('Meta-feature indicators calculated')

    logger.info('Completed')


if __name__ == '__main__':
    main(sys.argv[1:])
