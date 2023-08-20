import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from config import RETENTION_DATA_DIR, PLOT_DATA_DIR
import common as cm


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


def plot_retention(data, control):
    colors = ['#c4c4c4', '#f8b4c8', '#fb3b46', '#fd1dcc', '#fff59f']
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    groups = data.copy()
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


def plot_retention_statistics(stats):
    fig, ax = plt.subplots(1, 3, figsize=(15, 5), dpi=300)
    ax[0] = _retention_improvement_bar_plot(stats, ax[0])
    ax[1] = _bar_plot(stats, ax[1], 'intercept')
    ax[2] = _bar_plot(stats, ax[2], 'slope')
    plt.subplots_adjust(bottom=0.25)
    plt.tight_layout()
    return fig


def run_regression(df, statistics):
    assert statistics in {'improvement', 'retention'}
    y = np.log(df[statistics])
    X = np.array([np.log(df['day_n']), np.ones(len(df))]).T
    se = df['se'] / df[statistics]
    inv_cov = np.diag(1 / se ** 2)
    params_covariance = np.linalg.pinv(np.dot(np.dot(X.T, inv_cov), X))
    targets = np.dot(np.dot(X.T, inv_cov), y)
    params = np.dot(params_covariance, targets)
    out = {
        'slope': params[0],
        'slope_se': params_covariance[0, 0]**0.5,
        'intercept': params[1],
        'intercept_se': params_covariance[1, 1]**0.5,
        'd30': df[statistics].iloc[-5],
        'd30_se': df['se'].iloc[-5],
    }
    return out


def _format_retention_plot(ax):
    ax.set_xlabel('Day')
    ax.set_ylabel('Improvement Over GPT3.5 (%)')
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
    params = run_regression(stats, 'improvement')
    expected = np.exp(params['slope'] * np.log(stats['day_n']) + params['intercept'])
    ax.plot(stats['day_n'], expected, '--', color=color)
    return ax


def _bar_plot(stats, ax, stat_label):
    colors = ['#c4c4c4', '#f8b4c8', '#fb3b46', '#fd1dcc', '#fff59f']
    values = [model[stat_label] for model in stats.values()]
    labels = list(stats.keys())
    
    # sorted_indices = sorted(range(len(values)), key=lambda k: values[k], reverse=True)
    # dont sort, just hard code...
    sorted_indices = [3, 4, 1, 2, 0]
    values = [1 / (-values[i]) for i in sorted_indices]
    labels = [labels[i] for i in sorted_indices]
    bar_colors = [colors[i] for i in sorted_indices]

    ax.bar(labels, values, capsize=5, color=bar_colors, edgecolor='black')
    ax.set_title(stat_label.capitalize())
    ax.set_xticklabels(labels, rotation=45, ha='right')
    return ax


def _retention_improvement_bar_plot(stats, ax):
    colors = ['#c4c4c4', '#f8b4c8', '#fb3b46', '#fd1dcc', '#fff59f']
    values = [model['d30'] for model in stats.values()]
    labels = list(stats.keys())
    sorted_indices = [3, 4, 1, 2, 0]
    values = [values[i] for i in sorted_indices]
    values = [(v - values[-1]) * 100 / values[-1]  for v in values]
    labels = [labels[i] for i in sorted_indices]
    bar_colors = [colors[i] for i in sorted_indices]
    ax.bar(labels, values, capsize=5, color=bar_colors, edgecolor='black')
    ax.set_title('Retention Improvement Over Pygmalion %')
    ax.set_ylim(0, None)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    return ax


if __name__ == '__main__':
    data = cm.load_data(RETENTION_DATA_DIR)
    data = {name: add_retention(df) for name, df in data.items()}
    data['norm_0525_vicuna_13b_reward'] = normalise(
            data['0715_pyg'],
            data['0525_pyg_reward_baseline'],
            data['0525_vicuna_13b_reward']
            )
    data = {name: get_retention_stats(df) for name, df in data.items()}
    data = {
            'Pygmalion+ (6B)': data['0715_pyg'],
            'Vicuna+ (13B)': data['norm_0525_vicuna_13b_reward'],
            'ChaiLLM (6B)': data['0715_zl_v2e'],
            'Blended (13, 6, 6B)': data['0715_blended_v3_baseline'],
            'GPT3.5 (175B)': data['0715_azure_davinci'],
            }
    fig = plot_retention(data, 'GPT3.5 (175B)')
    fig.savefig(f'{PLOT_DATA_DIR}/retention.png')

    retention_stats = {name: run_regression(df, 'retention') for name, df in data.items()}
    fig = plot_retention_statistics(retention_stats)
    fig.savefig(f'{PLOT_DATA_DIR}/retention_stats.png')
