import numpy as np

def create_sequences(df, sensor_cols, window_size=30):
    X, y = [], []

    for unit in df['unit'].unique():
        unit_df = df[df['unit'] == unit].sort_values('cycle')

        for i in range(len(unit_df) - window_size):
            X.append(
                unit_df[sensor_cols].iloc[i:i+window_size].values
            )
            y.append(
                unit_df['RUL'].iloc[i+window_size-1]
            )

    return np.array(X), np.array(y)