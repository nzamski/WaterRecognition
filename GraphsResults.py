import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_performance(path, title):
    df = pd.read_csv(path)
    plot = sns.lineplot(data=df, x='Iteration', y='F1', hue='Hyperparameters')
    fig = plot.get_figure()
    plot.set(ylim=(0, 1))
    plot.set_title(title)
    plt.savefig(title+'.png', dpi=720)
    plt.show()


if __name__ == '__main__':
    plot_performance('Water_Bodies_Results.csv', 'F1_per_hyperparameters')
