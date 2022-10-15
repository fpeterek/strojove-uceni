import sys
import json

import matplotlib.pyplot as plt


def load_file(filename: str) -> tuple[list[float], list[float], list[int]]:
    xs = []
    ys = []
    clusters = []
    with open(filename) as file:
        cl = json.load(file)['clusters']
        for idx, cluster in enumerate(cl):
            for point in cluster:
                xs.append(point['x'])
                ys.append(point['y'])
                clusters.append(idx)

    return xs, ys, clusters


def plot_clusters(files: list[str], pair: tuple[str, str], fig, row):
    axes = [fig.axes[3*row], fig.axes[3*row + 1], fig.axes[3*row + 2]]

    for ax, clusters in zip(axes, [2, 3, 5]):
        file = [f for f in files if str(clusters) in f][0]
        xs, ys, clusters = load_file(file)
        ax.scatter(xs, ys, c=clusters)


def plot_group(group, files: list[str]):
    pairs = []
    for linkage in ['Complete', 'Single']:
        for metric in ['Euclidean', 'Manhattan']:
            pairs.append((linkage, metric))

    fig, axes = plt.subplots(len(pairs), 3, figsize=(30, 30))

    for row, (linkage, metric) in enumerate(pairs):
        row_files = [i for i in files if linkage in i and metric in i]
        plot_clusters(row_files, (linkage, metric), fig, row)

    fig.tight_layout()
    plt.savefig(f'{group}.png')


def last_index_of(string, substr) -> int:
    return len(string) - string[::-1].index(substr) - 1


def main(to_plot: list[str]) -> None:
    groups = set(map(lambda x: x[last_index_of(x, '/')+1:x.index('_')], to_plot))
    groups = [g for g in groups if 'annulus' in g]

    for group in groups:
        to_process = [it for it in to_plot if group in it]
        # print(to_process)
        plot_group(group, to_process)


if __name__ == '__main__':
    main(sys.argv[1:])
