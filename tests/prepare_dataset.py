#!/usr/bin/env python3
import argparse

import numpy as np
import sklearn.datasets as datasets

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_path", default="digits.txt", type=str, help="Dataset path")


def main(args):
    # print the digits dataset to a file
    data, target = datasets.load_digits(n_class=10, return_X_y=True)

    with open(args.dataset_path, "w") as dataset_file:
        for sample, sample_target in zip(data, target):
            dataset_file.write(" ".join([str(x) for x in sample]))
            dataset_file.write(" " + str(sample_target) + "\n")


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
