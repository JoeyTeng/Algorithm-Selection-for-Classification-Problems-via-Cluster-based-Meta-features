# @Author: Joey Teng <Toujour>
# @Date:   14-Nov-2017
# @Email:  joey.teng.dev@gmail.com
# @Filename: postprocess.py
# @Last modified by:   Toujour
# @Last modified time: 15-Nov-2017
import sys


def load(filename):
    import json

    return json.load(open(filename, 'r'))


def save(data, filename):
    import json

    return json.dump(data, open(filename, 'w'))


def interpret(clusters):
    return [(cluster["size"], cluster["radius"]) for cluster in clusters]


def plot(statistics, filename=None):
    import matplotlib.pyplot as plt
    import numpy as np

    max_size = 0
    for size, radius in statistics:
        max_size = max(max_size, size)

    sizes = np.zeros(max_size + 1, dtype=np.int64)
    radii = [[] for i in range(max_size + 1)]
    for size, radius in statistics:
        sizes[size] += 1
        radii[size].append(radius)

    for radius in radii:
        radius.sort()

    plt.figure(figsize=(9, 6))

    x = np.array(list(range(max_size + 1)))
    print(x)
    print(sizes, flush=True)
    plt.scatter(x, sizes, s=25, alpha=0.4, marker='o')
    # T:散点的颜色
    # s：散点的大小
    # alpha:是透明程度
    plt.show()


print(__name__, flush=True)
if __name__ == '__main__':
    print("INFO: Start", flush=True)
    clusters = load(sys.argv[1])
    statistics = interpret(clusters)
    plot(statistics)
    # save(clusters, sys.argv[2])
    print("INFO: Complete", flush=True)
    # paint(space, clusters)
