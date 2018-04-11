# @Author: Joey Teng
# @Email:  joey.teng.dev@gmail.com
# @Filename: test.py
# @Last modified by:   Joey Teng
# @Last modified time: 25-Jan-2018

import contextlib
import inspect
import json
import os

import nose.tools

import convex_hull_cluster
import spherical_cluster


STD_PATH = 'test/'
cluster = spherical_cluster


def get_path():
    # 0 is the current function
    # 1 in the father function
    return '{0}{1}'.format(STD_PATH,
                           inspect.stack()[1].function.replace('_', '.'))


def _test(path):
    with contextlib.suppress(FileNotFoundError):
        os.remove("{0}.clusters.json".format(path))
    cluster.main([path])
    nose.tools.assert_equal(
        json.load(open("{0}.clusters.json".format(path), 'r')),
        json.load(open("{0}.clusters.control.json".format(path))))
    os.remove("{0}.clusters.json".format(path))


def test_homo():
    path = get_path()
    _test(path)


def test_hetro():
    path = get_path()
    _test(path)


def test_hetro_size():
    path = get_path()
    _test(path)


def test_hetro_duplication():
    path = get_path()
    _test(path)
