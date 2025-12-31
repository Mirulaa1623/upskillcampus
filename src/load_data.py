import pandas as pd

def load_single_dataset(path):
    columns = ['unit', 'cycle', 'op1', 'op2', 'op3'] + \
              [f's{i}' for i in range(1, 22)]

    df = pd.read_csv(path, sep=' ', header=None)
    df = df.iloc[:, :-2]
    df.columns = columns

    return df


def load_multiple_datasets(paths):
    all_dfs = []
    unit_offset = 0

    for path in paths:
        df = load_single_dataset(path)

        # Make unit numbers unique across datasets
        df['unit'] = df['unit'] + unit_offset
        unit_offset = df['unit'].max()

        all_dfs.append(df)

    return pd.concat(all_dfs, ignore_index=True)