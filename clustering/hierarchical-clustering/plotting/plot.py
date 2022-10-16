import sys
import json

import matplotlib.pyplot as plt


def load_file(filename: str) -> tuple[list[float], list[float], list[int]]:
    xs = []
    ys = []
    cluster_labels = []
    with open(filename) as file:
        print(f'Loading {filename}')
        clusters = json.load(file)['clusters']
        for idx, cluster in enumerate(clusters):
            for point in cluster:
                xs.append(point['x'])
                ys.append(point['y'])
                cluster_labels.append(idx)

    return xs, ys, cluster_labels


def plot_clusters(files: list[str], clustering: tuple[str, str], fig, row):
    linkage, metric = clustering
    axes = [fig.axes[3*row], fig.axes[3*row + 1], fig.axes[3*row + 2]]

    for ax, num_clusters in zip(axes, [2, 3, 5]):
        file = [f for f in files if f[-6] == str(num_clusters)][0]
        xs, ys, cl_labels = load_file(file)
        ax.set_title(f'{linkage}, {metric}')
        ax.scatter(xs, ys, c=cl_labels)


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


def parse_group(filename: str) -> str:
    return filename[last_index_of(filename, '/')+1:filename.index('_')]


def group_inputs(inputs: list[str]):
    grouped = dict()

    for i in inputs:
        group = parse_group(i)
        lst = grouped.get(group, [])
        lst.append(i)
        grouped[group] = lst

    return grouped


def main(to_plot: list[str]) -> None:

    grouped = group_inputs(to_plot)

    for (group, files) in grouped.items():
        if len(files) != 12:
            continue
        # print(files)
        plot_group(group, files)


if __name__ == '__main__':
    main(sys.argv[1:])
