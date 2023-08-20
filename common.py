import os

import pandas as pd
import numpy as np


def load_data(data_dir):
    fnames = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    data = {f.split('.csv')[0]: pd.read_csv(os.path.join(data_dir, f)) for f in fnames}
    try:
        data = {name: df.drop(['Unnamed: 0'], axis=1) for name, df in data.items()}
    except KeyError:
        pass
    return data


def run_regression(df, statistics):
    df = df.dropna()
    assert statistics in {'improvement', 'retention', 'mean_engagement'}
    y = np.log(df[statistics])
    if statistics != 'mean_engagement':
        x = df['day_n']
    else:
        x = (df['upper_bounds'] + df['lower_bounds']) / 2
    X = np.array([np.log(x), np.ones(len(df))]).T
    se_col = 'se' if statistics != 'mean_engagement' else 'engagement_se'
    se = df[se_col] / df[statistics]
    inv_cov = np.diag(1 / se ** 2)
    params_covariance = np.linalg.pinv(np.dot(np.dot(X.T, inv_cov), X))
    targets = np.dot(np.dot(X.T, inv_cov), y)
    params = np.dot(params_covariance, targets)
    # Note: true intercept is needed for engagement, since the graph does not
    # start at time = 1, instead it starts off at its first bin. As a result
    # your actual intercept is the intercept + the slope * the first bin.
    # here I hacked it as I know my bin size is 1 day = 86400
    out = {
        'slope': params[0],
        'slope_se': params_covariance[0, 0]**0.5,
        'intercept': params[1],
        'intercept_se': params_covariance[1, 1]**0.5,
        'd30': df[statistics].iloc[-5],
        'd30_se': df[se_col].iloc[-5],
        'true_intercept': params[1] * params[0] * 43200,
    }
    return out


def bar_plot(stats, ax, stat_label, reverse=True):
    colors = ['#c4c4c4', '#f8b4c8', '#fb3b46', '#fd1dcc', '#fff59f']
    values = [model[stat_label] for model in stats.values()]
    labels = list(stats.keys())
    
    # sorted_indices = sorted(range(len(values)), key=lambda k: values[k], reverse=True)
    # dont sort, just hard code...
    sorted_indices = [3, 4, 1, 2, 0]
    if reverse:
        values = [1 / (-values[i]) for i in sorted_indices]
    else:
        values = [values[i] for i in sorted_indices]
    labels = [labels[i] for i in sorted_indices]
    bar_colors = [colors[i] for i in sorted_indices]

    ax.bar(labels, values, capsize=5, color=bar_colors, edgecolor='black')
    ax.set_title(stat_label.capitalize())
    ax.set_xticklabels(labels, rotation=45, ha='right')
    return ax
