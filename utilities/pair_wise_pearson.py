# @Author: Joey Teng
# @Email:  joey.teng.dev@gmail.com
# @Filename: pair_wise_pearson.py
# @Last modified by:   Joey Teng
# @Last modified time: 31-Jul-2018
import argparse
import collections
import csv

import scipy.stats.stats


def load(path):
    with open(path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)

        fieldnames = list(reader.fieldnames)[1:]  # Omit "Dataset"
        data = collections.OrderedDict([(key, list()) for key in fieldnames])
        for row in reader:
            for field in fieldnames:
                data[field].append(float(row[field]))

        print("Loaded: {}".format(path), flush=True)
        return data


def dump(path, data):
    print("Writing into csv file...", flush=True)
    with open("{}.paired_r.csv".format(path), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, dialect='excel')
        writer.writerows(data)

    print("Dump Completed.", flush=True)


def pair_wise_pearson(data):
    table = [["Quantity"] + list(data.keys())]
    for X in data.items():
        table.append(list([X[0]]))
        for Y in data.items():
            pearsonr = scipy.stats.pearsonr(X[1], Y[1])
            r_square = pearsonr[0] ** 2
            table[-1].append(r_square)
    return table


def main(args):
    path = args.i
    print(path, flush=True)
    data = load(path)
    output = pair_wise_pearson(data)
    dump(path, output)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calculate Pearson r square value pair-wisely"
    )
    parser.add_argument('-i', action='store', type=str,
                        help='Path to input csv file (with headers)')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
