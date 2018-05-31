# @Author: Joey Teng
# @Email:  joey.teng.dev@gmail.com
# @Filename: learning_rate_artificial_r_square_from_csv.py
# @Last modified by:   Joey Teng
# @Last modified time: 31-May-2018
import argparse
import collections
import csv
import os

import scipy.stats


def main(path):
    input_csv = path
    output_csv = "{}.with_r_square.csv".format(path[:-len('.csv')])
    data_csv = []
    fieldnames = []
    with open(input_csv, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        fieldnames = reader.fieldnames
        data_csv = [row for row in reader]

    fieldnames.append('Number of Separators')
    complexity_actual = []
    complexity_predicted = []
    for row in data_csv:
        tmp = row['Dataset Name'][len('artificial.'):]
        m = int(tmp[:tmp.find('.')])

        complexity_actual.append(float(row['Area Inverse']))
        complexity_predicted.append(m)

        row[fieldnames[-1]] = complexity_predicted[-1]

    pearsonr = scipy.stats.pearsonr(complexity_actual, complexity_predicted)
    r_square = pearsonr[0] ** 2

    data_csv.sort(key=lambda row: row[fieldnames[-1]])
    fieldnames[1], fieldnames[-1] = fieldnames[-1], fieldnames[1]

    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.DictWriter(
            csvfile, fieldnames=fieldnames, dialect='excel')
        writer.writeheader()
        writer.writerows(data_csv)
        writer.writerow(dict([
            (fieldnames[0], "r square of Inverse Area to Number of Separators"),
            (fieldnames[1], r_square)
        ]))


def parse_path():
    parser = argparse.ArgumentParser(
        description="Convert _value.json from graph plotting to csv.")
    parser.add_argument('-i', action='store', nargs='+', default=[],
                        help='Files that need to be processed')
    args = parser.parse_args()

    paths = sorted(args.i)
    for i in range(len(paths)):
        # Using relative path instead of absolute
        if not paths[i].startswith('/'):
            paths[i] = '{}/{}'.format(os.getcwd(), paths[i])

    return paths


if __name__ == '__main__':
    paths = parse_path()
    for path in paths:
        main(path)

    print("Program Ended", flush=True)
