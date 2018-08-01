# @Author: Joey Teng
# @Email:  joey.teng.dev@gmail.com
# @Filename: json_to_csv.py
# @Last modified by:   Joey Teng
# @Last modified time: 01-Aug-2018
import csv
import json
import sys


def main(path):
    print(path, flush=True)
    data = json.load(open(path))

    print("Writing into csv file...", flush=True)
    with open(
            "{}.csv".format(path[:-len('.json')]),
            'w', newline=''
            ) as csvfile:
        writer = csv.writer(csvfile, dialect='excel')
        writer.writerows(data)
    print("Completed.", flush=True)


if __name__ == '__main__':
    for path in sys.argv[1:]:
        main(path)
