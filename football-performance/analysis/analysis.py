import math

import pandas
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import sklearn.cluster
import sklearn.preprocessing


def load_df(path: str):
    return pandas.read_csv(path, sep=',')


def extract_comptype(df):
    is_cl = not df[df['League'] == 'Champions League'].empty

    return ['domestic', 'europe'][is_cl]


def histplot(df):

    df_num_only = df.select_dtypes(np.number)

    plots_in_row = 5
    figsize = (40, 50)

    _, n_cols = df_num_only.shape
    nrows = math.ceil(n_cols / plots_in_row)
    fig, axes = plt.subplots(nrows=nrows, ncols=plots_in_row, figsize=figsize)

    fig.tight_layout()

    for i, column in enumerate(df_num_only):
        row = i // plots_in_row
        col = i % plots_in_row
        plot = sns.histplot(data=df_num_only, x=column, ax=axes[row, col],
                            legend=False)
        plot.set_title(column)
        plot.set(xticklabels=[])
        plot.set(xlabel=None)
        plot.set(ylabel=None)

    comptype = extract_comptype(df)

    plt.savefig(f'plots/{comptype}_histplot.png')


def split_df(df):
    cl = df[df['League'] == 'Champions League'].copy()
    dl = df[df['League'] != 'Champions League'].copy()

    return dl, cl


def cluster(df, cluster_alg):
    df_num_only = df.select_dtypes(np.number)
    cluster_alg.fit(df_num_only)
    df = df.copy()
    df['cluster_id'] = cluster_alg.labels_

    grouped = df.groupby('cluster_id')

    return grouped


def print_clusters(df):
    for clusterid, cluster in df:
        print(clusterid)
        print('<details>')
        print(', '.join(cluster['Key']))
        print('</details>')


def cluster_df(df, alg, algname):
    print(f'----------------- {algname} -----------------')
    print(f'### {algname}')
    clusters = cluster(df, alg)
    print_clusters(clusters)


def nonna_attrs(df):
    total = df.shape[0]
    attrs = []
    for col in df.columns:
        notna_df = df[df[col].notna()]
        notna = notna_df.shape[0]
        na = total - notna
        # print(f'{col}: {total=}, {notna=}, {na=}')

        if not na:
            attrs.append(col)

    return attrs


def analyze_df(df, dbscan, kmeans):

    nonna = nonna_attrs(df)
    df = df[nonna]

    cluster_df(df, dbscan, 'DBSCAN')
    cluster_df(df, kmeans, 'KMeans')


def analyze_unprocessed(df):
    dbscan = sklearn.cluster.DBSCAN(eps=33, min_samples=5)
    kmeans = sklearn.cluster.KMeans(n_clusters=12, random_state=747)

    analyze_df(df, dbscan, kmeans)


def analyze_scaled_europe(df):
    dbscan = sklearn.cluster.DBSCAN(eps=1.05, min_samples=3)
    kmeans = sklearn.cluster.KMeans(n_clusters=12, random_state=747)

    analyze_df(df, dbscan, kmeans)


def analyze_scaled_domestic(df):
    dbscan = sklearn.cluster.DBSCAN(eps=0.73, min_samples=3)
    kmeans = sklearn.cluster.KMeans(n_clusters=12, random_state=747)

    analyze_df(df, dbscan, kmeans)


def scale(df, attr, scaler):
    df[attr] = scaler.fit_transform(df[[attr]])


def scale_attrs(df):
    num_only = df.select_dtypes(np.number)
    scaler = sklearn.preprocessing.MinMaxScaler()

    df = df.copy()

    for col in num_only.columns:
        scale(df, col, scaler)

    return df


def run_analysis():
    df = load_df('data/data.csv')
    domestic, europe = split_df(df)

    # histplot(europe)
    # histplot(domestic)

    nonna_attrs(europe)
    nonna_attrs(domestic)

    # analyze_unprocessed(europe)
    # analyze_unprocessed(domestic)

    scaled_europe = scale_attrs(europe)
    scaled_domestic = scale_attrs(domestic)

    analyze_scaled_europe(scaled_europe)
    analyze_scaled_domestic(scaled_domestic)
