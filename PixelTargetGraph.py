import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def rgb_graph(path):
    # extract data from CSV
    df = pd.read_csv(path)
    df = df.sample(n=1_000)
    agg_df = df.groupby(['r', 'g', 'b']).agg({'binary': np.mean, 'count': np.sum}).reset_index()
    sns.set(style="darkgrid")

    # create the plot figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x = agg_df['r']
    y = agg_df['g']
    z = agg_df['b']

    ax.set_xlabel("R")
    ax.set_ylabel("G")
    ax.set_zlabel("B")

    ax.scatter(x, y, z, cmap='coolwarm', c=agg_df['binary'], s=agg_df['count'])
    plt.show()


if __name__ == '__main__':
    rgb_graph('rgb_value_count.csv')
