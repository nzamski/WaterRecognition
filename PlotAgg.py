import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px


def rgb_graph(path):
    # extract data from CSV
    df = pd.read_csv(path)
    df['value'] = (df['value'] > 128).astype(int)  # redundant?
    agg_df = df.groupby(['r', 'g', 'b']).agg({'value': np.mean, 'count': np.sum}).reset_index()
    sns.set(style="darkgrid")

    # create the plot figure
    fig = plt.figure()

    '''ax = fig.add_subplot(111, projection='3d')  # redundant?
    
    x = agg_df['r']
    y = agg_df['g']
    z = agg_df['b']
    
    ax.set_xlabel("R")
    ax.set_ylabel("G")
    ax.set_zlabel("B")
    
    ax.scatter(x, y, z, cmap='coolwarm', c=agg_df['value'])
    
    plt.show()'''

    # present the plot
    fig = px.scatter_3d(agg_df, x='r', y='g', z='b', color='value')
    fig.show()


if __name__ == '__main__':
    rgb_graph('rgb_value_count.csv')
