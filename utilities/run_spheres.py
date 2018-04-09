# @Author: Joey Teng
# @Email:  joey.teng.dev@gmail.com
# @Filename: re_run.py
# @Last modified by:   Joey Teng
# @Last modified time: 09-Feb-2018

import sys
import os
import spherical_cluster


def main(paths):
    files = []
    for path in paths:
        files.extend([
            '{0}/{1}'.format(path, file.strip().strip('/'))
            for file in os.listdir(path) if file.find('.') == -1])

    for file in files:
        print(file)
        spherical_cluster.main([file])


if __name__ == '__main__':
    main(sys.argv[1:])
