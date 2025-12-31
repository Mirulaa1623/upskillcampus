from sklearn.preprocessing import StandardScaler

def add_rul(df):
    max_cycle = df.groupby('unit')['cycle'].max()
    df['RUL'] = df.apply(
        lambda row: max_cycle[row['unit']] - row['cycle'],
        axis=1
    )
    return df

def clean_and_scale(df):
    drop_sensors = ['s1', 's5', 's6', 's10', 's16', 's18', 's19']
    df = df.drop(columns=drop_sensors)

    sensor_cols = [c for c in df.columns if c.startswith('s')]

    scaler = StandardScaler()
    df[sensor_cols] = scaler.fit_transform(df[sensor_cols])

    return df, sensor_cols