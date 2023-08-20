import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from config import ENGAGEMENT_DATA_DIR, PLOT_DATA_DIR
import common as cm


def normalise(control, source, target):
    ratio = control['mean_engagement'] / source['mean_engagement']
    out = target.copy()
    out['mean_engagement'] = out['mean_engagement'] * ratio.mean()
    return out


def plot_engagement(data, control):
    colors = ['#c4c4c4', '#f8b4c8', '#fb3b46', '#fd1dcc', '#fff59f']
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    groups = data.copy()
    control_df = groups.pop(control)
    ax.axhline(1, color='k', label=f'{control} (baseline)')
    for idx, (name, df) in enumerate(groups.items()):
        color = colors[idx]
        stats = get_improvement_stats(df, control_df)
        ax.errorbar(stats['midpoint'], stats['ratio'], yerr=stats['se'], label=name, color=color)
        ax = _add_line_of_best_fit_plot(stats, ax, color)
    ax = _format_retention_plot(ax)
    plt.tight_layout()
    return fig


def _add_line_of_best_fit_plot(stats, ax, color):
    # not exact value, but close enough
    ratio = np.log(stats.dropna().ratio)
    x = np.log(stats.dropna().midpoint)
    slope, intercept = np.polyfit(x, ratio, 1)
    x = np.linspace(x.min(), x.max(), 100)
    ax.plot(np.exp(x), np.exp(intercept + slope * x), '--', color=color)
    return ax


def _format_retention_plot(ax):
    ax.set_xlabel('Days Since Experiment Start')
    ax.set_ylabel('Improvement Over GPT3.5 (%)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(None, 2160000)
    ax.set_xticks([86400, 86400 * 5, 86400 * 10, 86400 * 20])
    ax.set_xticklabels(['1 day', '5 days', '10 days', '20 days'])
    ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3])
    ax.set_yticklabels(["-50", "-40", "-30", "-20", "-10", '0', "10", "20", "30"])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.22), ncol=3)
    ax.spines[['right', 'top']].set_visible(False)
    return ax


def get_improvement_stats(df, control):
    ratio = df['mean_engagement'] / control['mean_engagement']
    se_group = df['engagement_se'] / df['mean_engagement']
    se_control = control['engagement_se'] / control['mean_engagement']
    se = (se_group**2 + se_control**2) ** 0.5 * ratio
    midpoint = (df['upper_bounds'] + df['lower_bounds']) / 2
    out = {
        'ratio': ratio,
        'midpoint': midpoint,
        'se': se
    }
    return pd.DataFrame(out)


if __name__ == '__main__':
    data = cm.load_data(ENGAGEMENT_DATA_DIR)
    data['norm_0525_vicuna_13b_reward'] = normalise(
            data['0715_pyg'],
            data['0525_pyg_reward_baseline'],
            data['0525_vicuna_13b_reward']
            )
    data = {
            'Pygmalion+ (6B)': data['0715_pyg'],
            'Vicuna+ (13B)': data['norm_0525_vicuna_13b_reward'],
            'ChaiLLM (6B)': data['0715_zl_v2e'],
            'Blended (13, 6, 6B)': data['0715_blended_v3_baseline'],
            'GPT3.5 (175B)': data['0715_azure_davinci'],
            }
    fig = plot_engagement(data, 'GPT3.5 (175B)')
    fig.savefig(f'{PLOT_DATA_DIR}/engagement.png')
