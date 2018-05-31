# @Author: Joey Teng
# @Email:  joey.teng.dev@gmail.com
# @Filename: learning_rate_json_to_csv.py
# @Last modified by:   Joey Teng
# @Last modified time: 08-May-2018


import argparse
import csv
import json
import os


def main(path):
    with open(path, 'r') as jsonfile:
        data = json.load(jsonfile)
    path_to_csv = "{}.csv".format(path[:-len('.json')])
    with open(path_to_csv, 'w', newline='') as csvfile:
        fieldnames = [
            "Dataset Name",
            "Coefficient",
            "R square",
            "Area Inverse"]
        writer = csv.DictWriter(
            csvfile, fieldnames=fieldnames, dialect='excel')
        writer.writeheader()
        for dataset in data:
            writer.writerow({
                fieldnames[0]:
                    dataset["dataset"],
                fieldnames[1]:
                    dataset["result"]["Random Forest"]["coefficients"][0],
                fieldnames[2]:
                    dataset["result"]["Random Forest"]["r_square"],
                fieldnames[3]:
                    dataset["result"]["Random Forest"]["area_inverse"]})


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
