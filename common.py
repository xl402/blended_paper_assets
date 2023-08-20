import os

import pandas as pd


def load_data(data_dir):
    fnames = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    data = {f.split('.csv')[0]: pd.read_csv(os.path.join(data_dir, f)) for f in fnames}
    try:
        data = {name: df.drop(['Unnamed: 0'], axis=1) for name, df in data.items()}
    except KeyError:
        pass
    return data
