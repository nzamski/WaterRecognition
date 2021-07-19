import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

if __name__ == '__main__':
    df = pd.read_csv('rgb_value_count.csv')
    df['value'] = (df['value'] > 128).astype(int)
    agg_df = df.groupby(['r', 'g', 'b']).agg({'value': np.mean, 'count': np.sum}).reset_index()
    sns.set(style="darkgrid")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x = agg_df['r']
    y = agg_df['g']
    z = agg_df['b']

    ax.set_xlabel("R")
    ax.set_ylabel("G")
    ax.set_zlabel("B")

    ax.scatter(x, y, z, cmap='coolwarm', c=agg_df['value'])

    plt.show()
