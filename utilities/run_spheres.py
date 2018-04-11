# @Author: Joey Teng
# @Email:  joey.teng.dev@gmail.com
# @Filename: re_run.py
# @Last modified by:   Joey Teng
# @Last modified time: 09-Feb-2018

import multiprocessing.pool
import sys
import os
import spherical_cluster


PROCESS_COUNT = int(os.cpu_count() / 2)


def run_task(file):
    print(file, flush=True)
    spherical_cluster.main([file])


def main(paths):
    files = []
    for path in paths:
        files.extend([
            '{0}/{1}'.format(path, file.strip().strip('/'))
            for file in os.listdir(path)
            if (file.find('.json') == -1
                and file.find('.log') == -1
                and file.find('.DS_Store') == -1)])
    files.sort()

    pool = multiprocessing.pool.Pool(PROCESS_COUNT)
    list(pool.map(run_task, files))
    pool.close()
    pool.join()


if __name__ == '__main__':
    main(sys.argv[1:])
