# @Author: Joey Teng <Toujour>
# @Date:   31-Jul-2017
# @Email:  joey.teng.dev@gmail.com
# @Filename: cluster.py
# @Last modified by:   Toujour
# @Last modified time: 12-Sep-2017

import pdb
import sys
import numbers
import collections


class Coordinate(object):
    def __init__(self, _x=None):
        self.x = _x

    def __add__(self, other):
        if (isinstance(other, type(self))):
            if (len(self.x) != len(other.x)):
                raise ValueError("Operands have different dimension: <lhs> {0}, <rhs> {1}".format(
                    len(self.x), len(other.x)))

            return Coordinate(_x=tuple([self.x[i] + other.x[i] for i in range(len(self.x))]))
        else:
            raise TypeError(
                "Operands have different type: <type> %r, <value> %r" % (type(value), value))

    def __mul__(self, other):
        if (isinstance(other, numbers.Number)):
            try:
                return Coordinate(_x=tuple(map(lambda x: x * other, self.x)))
            except AttributeError(e):
                print("Error: <type> {0}, <value> {1}".format(
                    type(self.x), self.x), flush=True)
                raise AttributeError(e)
        else:
            raise TypeError(
                "Operand is not a number: <type> %r, <value> %r" % (type(value), value))

    def __truediv__(self, other):
        if (isinstance(other, numbers.Number)):
            return Coordinate(_x=tuple(map(lambda x: x / other, self.x)))
        else:
            raise TypeError(
                "Operand is not a number: <type> %r, <value> %r" % (type(value), value))

    def __hash__(self):
        return hash("{0}".format(hash(self.x)))

    def __eq__(self, other):
        return hash(self) == hash(other)

    @property
    def x(self):
        if (self._x is None):
            raise TypeError("x coordinate is not defined")
        return self._x

    @x.setter
    def x(self, value):
        if (not isinstance(value, tuple)):
            raise TypeError(
                "Argument is not a tuple: <type> {0}, <value> {1}".format(type(value), value))
        if (isinstance(value[0], numbers.Number)):
            self._x = value
        else:
            raise TypeError(
                "Argument is not a number: <type> %r, <value> %r" % (type(value), value))

    def distance(self, other):
        try:
            other = other.coordinate
        except AttributeError:
            pass

        if (isinstance(other, self.__class__.__bases__[0])):
            if (len(self.x) != len(other.x)):
                raise ValueError("Operands have different dimension: <lhs> {0}, <rhs> {1}".format(
                    len(self.x), len(other.x)))

            return (sum([(self.x[i] - other.x[i])**2 for i in range(len(self.x))])**0.5)
        else:
            print("Error: {0} {1}".format(type(self),
                                          self.__calss__.bases__), flush=True)
            raise TypeError("Argument has a different type: <type> {0}, <value> {1}".format(
                type(other), other))


class Point(Coordinate):
    def __init__(self, _x, _label):
        super().__init__(_x=_x,)
        self.__label = _label

    def __add__(self, other):
        Coordinate = super().__add__(other)
        return Point(Coordinate.x, self.label)

    def __mul__(self, other):
        Coordinate = super().__mul__(other)
        return Point(Coordinate.x, self.label)

    def __truediv__(self, other):
        Coordinate = super().__truediv__(other)
        return Point(Coordinate.x, self.label)

    def __repr__(self):
        return {'dimension': self.x, 'label': self.label}

    @property
    def label(self):
        return self.__label


class Space(object):
    def __init__(self):
        self.point_set = []

    def __iter__(self):
        return self.point_set.__iter__()

    def append(self, value):
        self.point_set.append(value)

    def closest_to(self, point):
        return self.point_set


class Cluster(object):
    def __init__(self, space=None, _centroid=None, _radius=None):
        if (not isinstance(_centroid, Point)):
            raise TypeError("Wrong Type for Argument _centroid: <type> {0}, <value> {1}".format(
                type(_centroid), _centroid))

        self.centroid = _centroid
        self.radius = _radius
        self.set = {}
        self.update_set(space)
        self.homogenous = True
        self.checked = True

    def __getattr__(self, name):
        return getattr(self.centroid, name)

    def __len__(self):
        return len(self.set)

    def __or__(self, other):
        if (isinstance(other, type(self))):
            cluster = Cluster(_centroid=self.centroid, _radius=self.radius)
            cluster.set = {**self.set, **other.set}
            cluster.homogenous = False
            cluster.checked = False
            return cluster
        else:
            raise TypeError(
                "Operands have different type: <type> %r, <value> %r" % (type(value), value))

    def equals(self, other):
        if (isinstance(other, type(self))):
            return (self.set == other.set)
        else:
            raise TypeError(
                "Operands have different type: <type> %r, <value> %r" % (type(value), value))

    @property
    def label(self):
        if (self.homogenous):
            return self.centroid.label
        elif (self.checked):
            raise ValueError(
                "The cluster is not homogenous thus no valid label")

            return None
        else:
            self.homogenous = check_homogenous(self)
            self.checked = True
            return self.label

    @property
    def coordinate(self):
        return self.centroid

    def update_set(self, space=None):
        if (space == None):
            return

        for point in space.closest_to(self.centroid):
            distance = self.centroid.distance(point)
            if (distance <= self.radius):
                self.set[point] = 0  # distance
            else:
                pass  # break

    @property
    def size(self):
        return len(self)

    @property
    def values(self):
        return list(self.set.keys())


class Clusters(object):
    def __init__(self):
        self.cluster_set = []
        self.removed = []
        self.length = 0
        self.dirty = False

    def __getitem__(self, key):
        return self.cluster_set[key]

    def __setitem__(self, key, value):
        self.cluster_set[key] = value

    def __iter__(self):
        for i in range(len(self.cluster_set)):
            if (not self.removed[i]):
                yield self.cluster_set[i]
        return

    def __len__(self):
        return self.length

    def append(self, value):
        if (value == None):
            print ("Warning: Trying to append a None object", flush=True)

        self.cluster_set.append(value)
        self.removed.append(0)
        self.length += 1

        return len(self.cluster_set) - 1

    def fast_remove(self, key):
        if (self.removed[key] == 0):
            self.length -= 1
            self.removed[key] = 1
            self.dirty = True

    def prune(self):
        if (not self.dirty):
            return

        tmp_set = []
        tmp_removed = []
        for i in range(len(self.cluster_set)):
            if (not self.removed[i]):
                tmp_set.append(self.cluster_set[i])
                tmp_removed.append(0)
        self.cluster_set = tmp_set
        self.removed = tmp_removed
        self.dirty = False

    def sort(self):
        if (self.dirty):
            self.prune()
        self.cluster_set = sorted(self.cluster_set, key=(lambda x: x.label))

    def max_distance(self, coordinate):
        if (self.length == 0):
            return None

        for i in range(len(self.cluster_set)):
            if (not self.removed[i]):
                distance = coordinate.distance(
                    self.cluster_set[0]) - self.cluster_set[0].radius

        for i in range(len(self.cluster_set)):
            if (not self.removed[i]):
                distance = min(distance, self.cluster_set[i].distance(
                    coordinate) - self.cluster_set[i].radius)

        return distance


def check_homogenous(cluster):
    # pdb.set_trace()
    cluster.checked = True

    label = next(iter(cluster.values)).label
    for point in cluster.values:
        if (point.label != label):
            cluster.homogenous = False
            return False
    cluster.homogenous = True
    return True


def check_merge(space, clusters, lhs, rhs):
    # Return object
    if lhs.label != rhs.label:
        return None
    else:
        new_centroid = (lhs.coordinate * len(lhs) +
                        rhs.coordinate * len(rhs)) / len(lhs | rhs)
        radius = clusters.max_distance(new_centroid)
        cluster = Cluster(space, new_centroid, radius)

        if (not cluster.equals(lhs | rhs)):
            # print("Not equal")
            return None
        # else:
            # print("Equal")

        if (not check_homogenous(cluster)):
            # print("Not homogenous")
            return None
        # else:
            # print("Homogenous")
        return cluster


def main(path):
    space = Space()
    for line in open(path, 'r'):
        input = list(map(float, line.split(',')))
        dimension = tuple(input[:-1])
        label = int(input[-1])
        point = Point(dimension, label)
        space.append(point)
    print("INFO: Space initialized", flush=True)

    clusters = Clusters()
    for point in space:
        clusters.append(Cluster(space=space, _centroid=point, _radius=0))

    clusters.sort()
    print("INFO: Cluster initialized", flush=True)

    deque = collections.deque()
    for i in range(len(clusters)):
        deque.append(i)

    flag = True
    while flag and len(deque) > 1:
        flag = False
        length = len(deque)
        n = length
        while n:
            a = deque.pop()
            b = deque.pop()
            n -= 2
            merge = check_merge(space, clusters, clusters[a], clusters[b])
            if (merge):
                clusters.fast_remove(a)
                clusters.fast_remove(b)
                deque.append(clusters.append(merge))
            else:
                deque.append(a)
                deque.appendleft(b)
                n += 1

        if (length != len(deque)):
            flag = True

    clusters.prune()
    # pdb.set_trace()
    # print(check_merge(space, clusters, clusters[0], clusters[1]))
    print("INFO: Number of the clusters: {0}".format(len(clusters)))

    return (space, clusters)


def paint(space, clusters):
    import matplotlib.pyplot as plt
    import numpy as np

    t = np.arange(0, 2 * np.pi + 0.1, 0.1)
    for cluster in clusters:
        x = (cluster.radius * np.sin(t)) + cluster.centroid.x[0]
        y = (cluster.radius * np.cos(t)) + cluster.centroid.x[1]
        plt.plot(x, y, color=(abs(cluster.label), 0, abs(cluster.label - 1)))

    for point in space:
        plt.plot(point.x[0], point.x[1], marker=((int(point.label) * 'x')
                                                 or 'd'), color=(abs(point.label), 0, abs(point.label - 1)))

    plt.show()


def save(clusters, filename):
    import json

    clusters_data = []
    for cluster in clusters:
        clusters_data.append(
            {"centroid": cluster.centroid.__repr__(), "radius": cluster.radius})

    json.dump(clusters_data, open(filename, 'w'))


if __name__ == '__main__':
    print("INFO: Start", flush=True)
    space, clusters = main(sys.argv[1])
    save(clusters, sys.argv[2])
    print("INFO: Complete", flush=True)
    # paint(space, clusters)
