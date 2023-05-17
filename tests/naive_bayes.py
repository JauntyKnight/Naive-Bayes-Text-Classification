#!/usr/bin/env python3
import argparse
from math import log

import numpy as np
from scipy.stats import norm

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_path", default="digits.txt", type=str, help="Dataset path")
parser.add_argument("--classifier", default="bernoulli", type=str, help="Classifier type")
parser.add_argument("--test_size", default=0.5, type=lambda x: int(x) if x.isdigit() else float(x), help="Test size")
parser.add_argument("--alpha", default=1, type=float, help="Alpha parameter")

CLASSES = 10

def load_and_split_dataset(path, test_size):
    # Load the dataset from args.dataset_path
    data, targets = [], []
    with open(path, "r") as dataset_file:
        for line in dataset_file:
            line = line.split()
            targets.append(int(line[-1]))
            line.pop()
            data.append(np.array([float(x) for x in line]))
    
    # split the dataset into train and test
    train_data, test_data, train_target, test_target = [], [], [], []
    for i in range(len(data)):
        if i / len(data) < test_size:
            test_data.append(data[i])
            test_target.append(targets[i])
        else:
            train_data.append(data[i])
            train_target.append(targets[i])
    
    train_data = np.array(train_data)
    test_data = np.array(test_data)
    train_target = np.array(train_target)
    test_target = np.array(test_target)

    return train_data, train_target, test_data, test_target


def gaussianNB(train_data, train_target, test_data, test_target):
    means = np.zeros((CLASSES, train_data.shape[1]))
    vars = np.zeros((CLASSES, train_data.shape[1]))
    priors = np.zeros(CLASSES)
    for i in train_target:
        priors[i] += 1

    for i in range(CLASSES):
        priors[i] /= train_target.shape[0]

    # Training
    for k in range(means.shape[0]):
        for d in range(means.shape[1]):
            means[k][d] = np.sum([train_data[i][d] if train_target[i] == k else 0 for i in range(train_data.shape[0])]) / (priors[k] * train_target.shape[0])
            vars[k][d] = (np.sum([((train_data[i][d] - means[k][d]) ** 2) if train_target[i] == k else 0 for i in range(train_data.shape[0])]) / (priors[k] * train_target.shape[0]) + args.alpha) ** 0.5

    correctly_classified = 0

    # Prediction
    for sample in test_data:
        predictions = [np.product(norm.pdf(sample, means[k], vars[k])) * priors[k] for k in range(CLASSES)]
        if np.argmax(predictions) == test_target[correctly_classified]:
            correctly_classified += 1
    
    print(correctly_classified / test_target.shape[0])


def bernoulliNB(train_data, train_target, test_data, test_target):
    def binarize(x):
        return int(x >= 8)
    
    for i in range(train_data.shape[0]):
        for j in range(train_data.shape[1]):
            train_data[i][j] = binarize(train_data[i][j])
    
    for i in range(test_data.shape[0]):
        for j in range(test_data.shape[1]):
            test_data[i][j] = binarize(test_data[i][j])
    
    probs = np.zeros((CLASSES, train_data.shape[1]))

    priors = np.zeros(CLASSES)
    for i in train_target:
        priors[i] += 1

    for i in range(CLASSES):
        priors[i] /= train_target.shape[0]

    # Training
    for k in range(probs.shape[0]):
        for d in range(probs.shape[1]):
            probs[k][d] = (np.sum([train_data[i][d] if train_target[i] == k else 0 for i in range(train_data.shape[0])]) + args.alpha) / (priors[k] * train_target.shape[0] + 2 * args.alpha)

    biases = [log(priors[k]) + np.sum([log(1 - probs[k][d]) for d in range(train_data.shape[1])]) for k in range(CLASSES)]
    print(biases)

    weights = np.zeros((CLASSES, train_data.shape[1]))
    for k in range(weights.shape[0]):
        for d in range(weights.shape[1]):
            weights[k][d] = log(probs[k][d] / (1 - probs[k][d]))

    correclty_classified = 0

    # Prediction
    for sample in test_data:
        predictions = [biases[k] + sample.T @ weights[k] for k in range(CLASSES)]
        if np.argmax(predictions) == test_target[correclty_classified]:
            correclty_classified += 1
    
    print(correclty_classified / test_target.shape[0])


def multinomialNB(train_data, train_target, test_data, test_target):
    sums = np.zeros((CLASSES, train_data.shape[1]))

    priors = np.zeros(CLASSES)
    for i in train_target:
        priors[i] += 1

    for i in range(CLASSES):
        priors[i] /= train_target.shape[0]

    # Training
    for sample, sample_target in zip(train_data, train_target):
        sums[sample_target] += sample

    probs = np.zeros((CLASSES, train_data.shape[1]))

    for k in range(probs.shape[0]):
        for d in range(probs.shape[1]):
            probs[k][d] = (sums[k][d] + args.alpha) / (np.sum(sums[k]) + args.alpha * probs.shape[1])

    biases = [log(priors[k]) for k in range(CLASSES)]

    weights = np.zeros((CLASSES, train_data.shape[1]))
    for k in range(weights.shape[0]):
        for d in range(weights.shape[1]):
            weights[k][d] = log(probs[k][d])

    # Prediction
    for sample, sample_target in zip(test_data, test_target):
        predictions = [biases[k] + sample.T @ weights[k] for k in range(CLASSES)]
        print(*predictions)


def main(args: argparse.Namespace) -> float:
    if args.classifier == "gaussian":
        gaussianNB(*load_and_split_dataset(args.dataset_path, args.test_size))
    elif args.classifier == "bernoulli":
        bernoulliNB(*load_and_split_dataset(args.dataset_path, args.test_size))
    elif args.classifier == "multinomial":
        multinomialNB(*load_and_split_dataset(args.dataset_path, args.test_size))


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
