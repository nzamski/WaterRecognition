import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def comparison():
    otsu = pd.read_csv('Otsu_Results.csv')
    logit = pd.read_csv('test_per_pixel_Logit.csv')
    deep = pd.read_csv('Water_Bodies_Results.csv')

    # according to the original distribution in dataset
    baseline = 0.61992
    baseline_f1 = 0.76537

    deep = deep[deep['Model Name'] == 'Conv1']
    deep = deep[deep['Hidden Layer Size'] == 3000]
    deep = deep[deep['Activation Function'] == 'leaky_relu']
    deep = deep[['Iteration', 'F1', 'Model Name']]

    otsu = pd.DataFrame({'Iteration': [i + 1 for i in range(0, 10)],
                         'F1': [otsu['F1'].median() for _ in range(0, 10)]})

    otsu['Model Name'] = "Otsu's Method (median)"
    logit['Model Name'] = 'Logistic Regression'
    deep['Model Name'] = 'Conv1 (Leaky ReLU, 3000²)'
    df = pd.concat([otsu, logit, deep]).set_index([[i for i in range(0, 30)]])

    plot = sns.lineplot(data=df, x='Iteration', y='F1', hue='Model Name')
    plot.axhline(baseline_f1, ls='--', c='r', label='Baseline')
    plot.set(ylim=(0.7, 1))
    plot.set_title('All 3 methods: F1-score for best models')
    plt.savefig('comparison.png', dpi=720)
    plt.show()


def method_plot():
    otsu = pd.read_csv('Otsu_Results.csv')
    sgd = pd.read_csv('test_per_pixel_SGD.csv')
    logit = pd.read_csv('test_per_pixel_Logit.csv')
    deep = pd.read_csv('Water_Bodies_Results.csv')

    # according to the original distribution in dataset
    baseline = 0.61992
    baseline_f1 = 0.76537

    # deep.loc[deep['Activation Function'] == 'leaky_relu', 'Activation Function'] = 'Leaky ReLU'
    # deep.loc[deep['Activation Function'] == 'relu', 'Activation Function'] = 'ReLU'

    plot = sns.lineplot(data=sgd, x='Iteration', y='F1')
    plot.axhline(baseline_f1, ls='--', c='r', label='Baseline')
    # plot.set(ylim=(0, 1))
    plot.set_title('SGD: F1-score over iterations')
    plt.savefig('plot.png', dpi=720)
    plt.show()


if __name__ == '__main__':
    method_plot()
