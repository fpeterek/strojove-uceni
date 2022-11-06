import sys
import os
from pathlib import Path

import matplotlib.pyplot as plt


def plot(infile):
    train = []
    test = []
    depths = []

    no_suffix = os.path.splitext(infile)[0]
    title = Path(infile).stem

    with open(infile) as f:
        lines = f.readlines()

    for line in lines:
        split = line.split(': (')  # )

        depth = int(split[0])
        acc = split[1][:-2].split(',')

        depths.append(depth)
        train.append(float(acc[0]))
        test.append(float(acc[1]))

    plt.figure(layout='constrained')
    plt.plot(depths, train, label='Training')
    plt.plot(depths, test, label='Validation')
    plt.legend()
    plt.title(title)

    plt.xlabel('Tree Depth')
    plt.ylabel('Accuracy')

    plt.savefig(f'{no_suffix}.png')


if __name__ == '__main__':
    files = sys.argv[1:]
    for file in files:
        plot(file)
