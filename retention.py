import os

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from config import RETENTION_DATA_DIR, PLOT_DATA_DIR


def load_data(data_dir):
    fnames = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    data = {f.split('.csv')[0]: pd.read_csv(os.path.join(data_dir, f)) for f in fnames}
    data = {name: df.drop(['Unnamed: 0'], axis=1) for name, df in data.items()}
    return data


def add_retention(df):
    df['retention'] = df['retained'] / df['users']
    return df


def normalise(control, source, target):
    ratio = control['retention'] / source['retention']
    out = target.copy()
    out['retention'] = out['retention'] * ratio.mean()
    out['retained'] = out['users'] * out['retention']
    out['retained'] = out['retained'].astype(int)
    return out


def get_retention_stats(df):
    rate = (df['retained'] / df['users']).values
    se = (rate * (1 - rate) / df['users']).values ** 0.5
    return pd.DataFrame({'day_n': df['day_n'], 'retention': rate, 'se': se})


def get_improvement_stats(df, control):
    improvement = df['retention'] / control['retention']
    se_group = df['se'] / df['retention']
    se_control = control['se'] / control['retention']
    se = (se_group**2 + se_control**2) ** 0.5 * improvement
    out = {
        'day_n': df['day_n'],
        'improvement': improvement,
        'se': se
    }
    return pd.DataFrame(out)


def plot_retention(groups, control):
    colors = ['#c4c4c4', '#f8b4c8', '#fb3b46', '#fd1dcc', '#fff59f']
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    control_df = groups.pop(control)
    ax.axhline(1, color='k', label=f'{control} (baseline)')
    for idx, (name, df) in enumerate(groups.items()):
        color = colors[idx]
        stats = get_improvement_stats(df, control_df)
        ax.errorbar(stats['day_n'], stats['improvement'], yerr=stats['se'], label=name, color=color)
        ax = _add_line_of_best_fit_plot(stats, ax, color)
    ax = _format_retention_plot(ax)
    plt.tight_layout()
    return fig


def _format_retention_plot(ax):
    ax.set_xlabel('Day')
    ax.set_ylabel('Retention Improvement (%)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30])
    ax.set_xticklabels([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30])
    ax.set_yticks([0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4])
    ax.set_yticklabels(["-30", "-20", "-10", '0', "10", "20", "30", "40"])
    ax.set_xlim(1, 30.2)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.22), ncol=3)
    ax.spines[['right', 'top']].set_visible(False)
    return ax


def _add_line_of_best_fit_plot(stats, ax, color):
    params = _run_regression(stats)
    expected = np.exp(params['slope'] * np.log(stats['day_n']) + params['intercept'])
    ax.plot(stats['day_n'], expected, '--', color=color)
    return ax


def _run_regression(df):
    y = np.log(df['improvement'])
    X = np.array([np.log(df['day_n']), np.ones(len(df))]).T
    se = df['se'] / df['improvement']
    inv_cov = np.diag(1 / se ** 2)
    params_covariance = np.linalg.pinv(np.dot(np.dot(X.T, inv_cov), X))
    targets = np.dot(np.dot(X.T, inv_cov), y)
    params = np.dot(params_covariance, targets)
    out = {
        'slope': params[0],
        'slope_se': params_covariance[0, 0]**0.5,
        'intercept': params[1],
        'intercept_se': params_covariance[1, 1]**0.5,
    }
    return out



if __name__ == '__main__':
    data = load_data(RETENTION_DATA_DIR)
    data = {name: add_retention(df) for name, df in data.items()}
    data['norm_0525_vicuna_13b_reward'] = normalise(
            data['0715_pyg'],
            data['0525_pyg_reward_baseline'],
            data['0525_vicuna_13b_reward']
            )
    data = {name: get_retention_stats(df) for name, df in data.items()}
    groups = {
            'Pygmalion+ (6B)': data['0715_pyg'],
            'Vicuna+ (13B)': data['norm_0525_vicuna_13b_reward'],
            'ChaiLLM (6B)': data['0715_zl_v2e'],
            'Blended': data['0715_blended_v3_baseline'],
            'GPT3.5 (175B)': data['0715_azure_davinci'],
            }
    fig = plot_retention(groups, 'GPT3.5 (175B)')
    fig.savefig(f'{PLOT_DATA_DIR}/retention.png')
