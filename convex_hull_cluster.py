# @Author: Joey Teng <Toujour>
# @Date:   20-Nov-2017
# @Email:  joey.teng.dev@gmail.com
# @Filename: convex_hull_cluster.py
# @Last modified by:   Toujour
# @Last modified time: 24-Jan-2018
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
import functools
import itertools
import json
import logging
import logging.handlers
import math
import multiprocessing.pool
import os
import queue
import sys

import numpy
import scipy.special

PROCESS_COUNT = int(os.cpu_count() / 2)


def _tree():
    """Define a recursive structure of collection.defaultdict(self)."""
    return collections.defaultdict(_tree)


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

    return logger


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


def signed_volume(vertices):
    """Calculate the signed volume of n-dimensional simplex.

    The simplex is defined by (n + 1) vertices
    Reference:
        Wedge Product: http://mathworld.wolfram.com/WedgeProduct.html

    Args:
        vertices (Vertices): Define the n-d simplex.

    Returns:
        tuple: (
            sign (float):
                -1, 0 or 1, the sign of the signed volume,
            logvolume (float):
                The natural log of the absolute value of the volume)

        If the signed volume is zero, then sign will be 0
            and logvolume will be -Inf.
        In all cases, the signed volume is equal to sign * np.exp(logvolume)

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
    """Calculte the squared area of the n-1-d simplex.

    Calculate the squared area of (n - 1)-dimensional simplex defined by
        n vertices in n-dimensional space
    Reference:
        Wedge Product: http://mathworld.wolfram.com/WedgeProduct.html


    Args:
        vertices (Vertices): Define the n-1-d simplex

    Returns:
        float: The natural log of the squared area of the simplex

    Reference:
        From scipy manual
            https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.linalg.slogdet.html#numpy.linalg.slogdet

    """
    dimension = len(vertices[0])
    matrix = numpy.matrix(
        numpy.stack(vertices[1:]) +
        numpy.array(vertices[0]) *
        numpy.ones((len(vertices) - 1, dimension)) * -1)
    logvolume = numpy.linalg.slogdet(matrix * matrix.T)[1]  # sign, logvolume
    return logvolume


def check_inside(face, instance, edge=None, area=None):
    """Check if the instance given is at the inner side of the face.

    Args:
        face (Vertices): [description]
        instance (Vertex): [description]
        edge (Vertices, optional): Defaults to None.
            By default, edge = face[:-1]
            Used to calculate the area and
                thus check when instance is on the same plane with the face.
        area (float, optional): Defaults to None.
            By default, area = squared_area(face)

    Returns:
        tuple: (
            inside (bool),
            new face generated with (edge + pivot) (Vertices),
            new squared_area calculated using new face (float))

    """
    edge = edge or face[:-1]
    area = area or squared_area(face)

    sign, logvolume = signed_volume(form_face(face, instance))
    _face = form_face(edge, instance)
    _area = squared_area(_face)
    if ((numpy.isclose([numpy.exp(logvolume)], [0]) and _area > area)
       or sign < 0):
        # outside
        return (False, _face, _area)
    return (True, _face, _area)


def check_inside_hull(hull, instance):
    """Check if the instance given is inside the hull.

    Args:
        hull (list): Faces on the hull
        instance (Vertex): [description]

    Returns:
        bool: If the instance is inside the hull

    """
    for face in hull:
        if not check_inside(face=face, instance=instance)[0]:
            return False
    return True


def check_homogeneity(impurities, hull, used_pivots):
    """Check if the hull is homogeneous.

    Args:
        impurities (Vertices): Instances with different label
        hull (list): all the faces of the hull
        used_pivots (set): [description]

    Returns:
        bool: If the convex hull have homogeneity

    """
    for instance in impurities:
        if instance in used_pivots:
            continue
        if check_inside_hull(hull, instance):
            return False

    return True


def pivot_on_edge(instances, edge, used_pivots):
    """Search for the next best possible vertex on the hull.

    Homogeneity of the hull may not be maintained.

    Args:
        instances (Vertices): [description]
        edge (Vertices): [description]
        used_pivots (set): [description]

    Recieve:
        Homogeneity (bool): If the choice of the vertex will maintain
            the homogeneity of the hull

    Yields:
        tuple:
            (None, False): No vertex is found
            (pivot (Vertex), homogeneity (bool)): A candidate is returned,
                with the side-effect of homogeneity of the hull
            (pivot (Vertex)): A candidate is found and
                checking of homogeniety is requested

    """
    vertices_in_edge = set(edge)
    index = 0
    length = len(instances)
    while index < length and instances[index] in used_pivots:
        index += 1

    if index == length:
        yield (None, False)  # Not found
        return

    homo = {}
    homo['pivot'] = instances[index]
    homo['face'] = form_face(edge, homo['pivot'])
    homo['area'] = squared_area(homo['face'])

    homogeneity = False
    check = yield (homo['pivot'], )
    if check:
        homogeneity = True

    for instance in instances:
        if instance in vertices_in_edge:
            # Skip all used pivots in edge to prevent self-orientating
            # Skip all instances labelled differently
            # Homogeneity test is checked every round
            continue

        current = {}
        current['pivot'] = instance
        inside, current['face'], current['area'] = check_inside(
            homo['face'], current['pivot'],
            edge=edge, area=homo['area'])

        if not inside:
            check = yield (current['pivot'], )
            if check:
                # update
                homo = current
                homogeneity = True

    yield (homo['pivot'], homogeneity)
    return


def find_next_pivot(instances, hull, edge,
                    used_pivots, edge_count, impurities):
    """Find next available vertex while ensure the homogeneity.

    Iteratively call pivot_on_edge() and check_homogeneity()
       to find the next available vertex on the hull.

    Args:
        instances (Vertices):
        hull (list): Faces of the hull
        edge (Vertex):
        used_pivots (set):
        edge_count (list):
        impurities (Vertices):

    Returns:
        pivot (Vertex):
        found (bool):

    """
    find_pivot = pivot_on_edge(instances, edge, used_pivots)
    pivot = next(find_pivot)
    while len(pivot) == 1:
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
        check['_edges'] = [
            tuple(sort_vertices(edge))
            for edge in itertools.combinations(
                check['_face'], len(check['_face']) - 1)]
        for _edge in check['_edges']:
            edge_count[_edge] += 1

        check['number of face added'] = close_up_hull(
            hull, edge_count, used_pivots)

        check['homogeneity'] = check_homogeneity(
            impurities, hull, used_pivots)
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
    """Form face by appending pivot and convert it into a tuple.

    Args:
        edge (Vertices): [description]
        pivot (Vertex): [description]

    Returns:
        tuple: Face formed

    """
    return tuple(list(edge) + [pivot])


def close_up(edge_count, used_pivots):
    """Provide faces required to close up the hull with existing vertices.

    Args:
        edge_count (dict): [description]
        used_pivots (set): [description]

    Returns:
        list: Faces required.

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
            updated = False
            for edge in lazy_update:  # = .keys()
                if lazy_update[edge] + edge_count[edge] == 1:
                    edges.append(edge)
                    lazy_update[edge] = 2  # Avoid duplicated edges
                    updated = True
            if not updated:
                break
            continue

        face = list(vertices)
        for pivot in used_pivots:  # = .keys()
            if pivot not in vertices:
                if not check_inside(face, pivot)[0]:
                    # det(A) = -det (B) if two cols swap (odd and even)
                    face[-1], face[-2] = face[-2], face[-1]
                break
        else:
            # This edge is the first edge
            return []

        faces.append(tuple(face))
        for edge in itertools.combinations(tuple(face), len(face) - 1):
            lazy_update[tuple(sort_vertices(edge))] += 1

    return faces


def close_up_hull(hull, edge_count, used_pivots):
    """Close up the hull.

    Second stage.
    Add all remaining faces into the hull to form
        a closed simplicial complex

    Args:
        hull (list): All faces of the hull.
        edge_count (dict): [description]
        used_pivots (set): [description]

    Returns:
        int: Number of face added

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
    """Call wrapped sorting function.

    A wrapper of sorting function
    Using buitin sorted() for now

    Args:
        same as the wrapped function

    Returns
        same as the wrapped function

    Raises:
        same as the wrapped fucntion

    """
    return sorted(*args, **kwargs)


def qsort_partition(data, target=1, lhs=0, rhs=None):
    """Find the smallest [target] values in the [data] using [comp] as __lt__.

    Complexity: O(n)

    Args:
        data (Vertices): A list of vertex in tuple type
        target (int, optional): Defaults to 1.
            [terget] smallest values will be returned.
        lhs (int, optional): Defaults to 0. Lowest index
        rhs (int, optional): Defaults to None. Highest index + 1
        comp (func, Currently not supported): Defaults to __builtin__.__lt__.
            Cumstomised function used for comparing

    Returns:
        list: [target] shallow copies of Vertex

    """
    # comp is Partially supported: only used in partitioning
    # but not in sorting return values
    # BUG: Work around instead for now
    # comp = (lambda x, y: x < y)

    data = list(set(data))  # Remove repeated vertices

    # BUG: Work around instead for now

    # lhs = lhs or 0
    # rhs = len(data) - 1  # Since [data] is updated
    # position = -1

    # while position != target:
    #     if position < target:
    #         lhs = position + 1
    #     elif position > target:
    #         rhs = position - 1

    #     pivot = data[rhs]
    #     index = lhs
    #     for i in range(lhs, rhs + 1):
    #         if comp(data[i], pivot):
    #             data[i], data[index] = data[index], data[i]
    #             index += 1
    #     data[rhs], data[index] = data[index], data[rhs]
    #     position = index  # Return value
    # return sort_vertices(data[:target])

    return sort_vertices(data)[:target]


def initialize_hull(instances, impurities):
    """Initialize the hull by obtain the first face of the hull.

    face: a n-1-d structure

    Args:
        instances (Vertices): Instances with same label
        impurities (Vertices): Instances with different label

    Returns:
        tuple:
            dimension (int): Dimension of the space, n
            face (tuple): The face obtained
                (Vertex, ...)
            used_pivots (set): The set of used instances on the hull
                set{Vertex}
            edge_count (dict): Counting of how many times an edge is used
                {edge (Vertices): times (int)}

    """
    dimension = len(instances[0])
    edge = qsort_partition(instances, target=dimension - 1)
    used_pivots = set(edge)
    edge_count = collections.defaultdict(int)  # default = 0
    face = edge
    if len(edge) == dimension - 1:
        pivot, found = find_next_pivot(
            instances, [], edge, used_pivots, edge_count, impurities)
        if found:
            face = form_face(edge, pivot)
            used_pivots.add(pivot)
    return (dimension, tuple(face), used_pivots, edge_count)


def queuing_face(face, _queue, edge_count):
    """Push all the possible edges (n-2-d structure) into the queue.

    Edges are obtained by making combinations.

    No edge will join the queue more than once.

    Gurantee the order that the later one in the face
        will be excluded first in combinations.

    Args:
        face (Vertices): A face made of many vertices (n-1)
        _queue (Queue): Target queue which supports .push()
        edge_count (dict): Counting of how many times an edge is used
            {edge (Vertices): times (int)}

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


def gift_wrapping(instances, impurities, logger):
    """Use modified gift-wrapping method for convex hull building.

    Two stages: Finding new vertex & Close-up

    Args:
        instances (Vertices): List of instances with same label
        impurities (Vertices): List of instances with different label

    Returns:
        dict:
            {
                "faces": All the faces,
                    list: [face]
                "vertices": All the vertices
                    dict: {Vertex: True}
                "dimension": Dimension of the hull
                    int: len(face)
            }

    """
    dimension, face, used_pivots, edge_count = initialize_hull(
        instances, impurities)
    _queue = queue.LifoQueue()
    if len(face) == dimension:
        queuing_face(face, _queue, edge_count)

    hull = []
    hull.append(face)
    vertices = [coordinate for coordinate in face]

    slices = 8
    all_instances = instances
    instances = [
        all_instances[
            int(len(all_instances) * i / slices):
            int(len(all_instances) * (i + 1) / slices)]
        for i in range(slices)]
    # First stage: find all new pivots
    while not _queue.empty():
        edge = _queue.get()
        if edge_count[edge] > 1:
            continue

        pool = multiprocessing.pool.Pool(PROCESS_COUNT)
        func = functools.partial(
            find_next_pivot,
            hull=hull, edge=edge, used_pivots=used_pivots,
            edge_count=edge_count, impurities=impurities)
        result = pool.map(func, instances)
        result = list(map(func, instances))
        pool.close()
        pool.join()

        not_found = [i[0] for i in enumerate(result) if i[1][0] is None]
        candidate = [element[0] for element in result if element[0]]
        pivot, found = func(candidate)
        if found:
            pivot, found = func(list(itertools.chain(
                *[instances[i] for i in not_found], [pivot])))
        if not found:
            continue

        face = form_face(edge, pivot)
        vertices.append(pivot)
        used_pivots.add(pivot)
        hull.append(face)
        queuing_face(face, _queue, edge_count)

    logger.debug("gift_wrapping: First stage complete. Starting second.")
    # Second stage: close up the hull
    if dimension < len(used_pivots):
        close_up_hull(hull, edge_count, used_pivots)
    logger.debug("gift_wrapping: Second stage complete.")
    return {
        "faces": hull,
        "vertices": used_pivots,
        "dimension": dimension}


def map_generate_tuple(*args):
    """Generate a tuple with the results from the func.

    Used to assist dict(), map() to generate a dictionary.

    Args:
        *args (list): [0]:(
            key (immutable): key of the generated dict,
            func (function): function to be called,
            arg (tuple): arguments for func)

    Returns:
        tuple: (key, func(*arg))

    """
    key, func, arg = args[0][0], args[0][1], args[0][2]
    return (key, func(*arg))


def clustering(dataset, logger):
    """Calculate all convex hulls.

    All hulls will be pure(only contains data points with same label)

    Args:
        dataset (list]): All the instances in the space with label
            list of dict objects:
            [Point, ...]
        logger (logger): logger for logging

    Returns:
        dict: Clusters obtained separated by labels
            label: clusters (list of dict objects)
                [{
                'vertices' (list): Turning instances on the hull
                    [Vertex, ...],
                'points': Instances in the hull. Vertices are excluded
                    [Vertex, ...]
                'size' (int): Number of instances covered by the hull
                    len(['vertices']) + len(['points']),
                'volume': The volume of the hull
                    float(optional)
                }, ...]

    """
    all_instances = dataset
    meta_dataset = collections.defaultdict(list)
    for instance in all_instances:
        meta_dataset[instance['label']].append(instance['coordinate'])

    tasklist = map(
        lambda item, meta_dataset=meta_dataset, logger=logger: (
            item[0],
            clustering_by_label,
            (item[1], item[0], meta_dataset, logger)), meta_dataset.items())

    # pool = multiprocessing.pool.Pool(PROCESS_COUNT)
    # clusters = dict(pool.map(map_generate_tuple, tasklist))
    clusters = dict(map(map_generate_tuple, tasklist))
    # pool.close()
    # pool.join()

    return clusters


def clustering_by_label(instances, label, meta_dataset, logger):
    """Obtain all possible clusters with given label.

    Args:
        instances (Vertices): all instances with given label
        label (label): label
        meta_dataset (meta_dataset): dict of the whole dataset
        logger (logger): logger inherited

    Returns:
        list: list of all clusters obtained

    """
    clusters = []
    impurities = {
        item[0]: item[1]
        for item in meta_dataset.items() if item[0] != label}
    impurities = list(itertools.chain(*impurities.values()))

    while instances:
        # List is not empty
        cluster = gift_wrapping(instances, impurities, logger)

        found = cluster['dimension'] < len(cluster['vertices'])
        _dataset = []
        vertices = []
        points = []
        for vertex in instances:
            if vertex in cluster['vertices']:
                vertices.append(vertex)
            else:
                if found and check_inside_hull(cluster['faces'], vertex):
                    points.append(vertex)
                else:
                    _dataset.append(vertex)

        if found:
            volume = round(calculate_volume(cluster['faces']), 15)
        elif len(cluster['faces'][0]) > 1:
            volume = round(squared_area(cluster['faces'][0]), 15)
        else:
            volume = 0.0

        instances = _dataset
        clusters.append({'vertices': vertices,
                         'points': points,
                         'size': len(vertices) + len(points),
                         'volume': volume})

        logger.info(
            'Clustering: %d clusters found, '
            '%d/%d instance processed for label %r',
            len(clusters), len(meta_dataset[label]) - len(instances),
            len(impurities) + len(meta_dataset[label]), label)

    return clusters


def calculate_volume(hull):
    """Calculate the volume of a convex hull.

    Args:
        hull (list): All faces in the hull.

    Returns:
        float: Volume calculated.

    """
    origin = hull[0][0]
    volume = 0.0
    for face in hull:
        logvolume = signed_volume(form_face(face, origin))[1]
        volume += numpy.exp(logvolume)
    # n-dimensional simplex = det / n!
    volume /= scipy.special.factorial(len(origin))

    return volume


def size_versus_number_of_clusters(clusters):
    """Calculate the number of clusters respect to each size.

    Args:
        clusters (list): list of clusters

    Returns:
        dict: {size (int): quantity (int), ...}

    """
    stats = collections.defaultdict(int)  # default = 0
    for cluster in clusters:
        # initial quantity is 0
        stats[cluster['size']] += 1
    return stats


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


def centroid(clusters):
    """Calculate the centroid of the vertices on the convex hulls.

    Inner instances are excluded.

    Args:
        clusters (list): list of clusters

    Returns:
        list: [vertex, ...]

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
        float: interval
            range / slots
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


def meta_features(clusters):
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
    """Start main function here."""
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
