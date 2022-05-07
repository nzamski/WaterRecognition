import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# according to the original distribution in dataset
baseline = 0.61992
baseline_f1 = 0.76537


def comparison():
    otsu = pd.read_csv('Otsu_Results.csv')
    logit = pd.read_csv('Logit_Results.csv')
    deep = pd.read_csv('Deep_Results.csv')

    deep = deep[deep['Model Name'] == 'Conv1']
    deep = deep[deep['Hidden Layer Size'] == 2000]
    deep = deep[deep['Activation Function'] == 'relu']
    deep = deep[['Iteration', 'F1', 'Model Name']]
    deep.loc[deep['Iteration'] == 1, 'F1'] = 0

    otsu = pd.DataFrame({'Iteration': [i + 1 for i in range(0, 10)],
                         'F1': [otsu['F1'].median() for _ in range(0, 10)]})

    otsu['Model Name'] = "Otsu's Method (median)"
    logit['Model Name'] = 'Logistic Regression'
    deep['Model Name'] = 'LochNet (ReLU)'
    df = pd.concat([otsu, logit, deep]).set_index([[i for i in range(0, 30)]])
    df.rename(columns={'Model Name': 'Method'}, inplace=True)

    plot = sns.lineplot(data=df, x='Iteration', y='F1', hue='Method')
    plot.axhline(baseline_f1, ls='--', c='k', label='Baseline')
    plot.set(ylim=(0, 1))
    plot.set_title('All 3 methods: F1-score over the iterations')
    plt.savefig('comparison.svg', dpi=2540)
    plt.show()


def method_plot():
    otsu = pd.read_csv('Otsu_Results.csv')
    sgd = pd.read_csv('SGD_Results.csv')
    logit = pd.read_csv('Logit_Results.csv')
    deep = pd.read_csv('Deep_Results.csv')

    deep.loc[deep['Activation Function'] == 'leaky_relu', 'Activation Function'] = 'Leaky ReLU'
    deep.loc[deep['Activation Function'] == 'relu', 'Activation Function'] = 'ReLU'

    loch_net = deep.loc[(deep['Model Name'] == 'Conv1') & (deep['Hidden Layer Size'] == 2000)]

    logit['Model Name'] = 'Logistic Regression'
    sgd['Model Name'] = 'Stochastic Gradient Descent'
    machine = pd.concat([sgd, logit]).set_index([[i for i in range(0, 20)]])

    plot = sns.lineplot(data=machine, x='Iteration', y='F1', hue='Model Name')
    plot.axhline(baseline_f1, ls='--', c='k', label='Baseline')
    plot.set(ylim=(0, 1))
    plot.set_title('Machine Learning: F1-score over the iterations')
    plt.savefig('machine.svg', dpi=2540)
    plt.show()


def time_plot():
    df = pd.DataFrame(data=[["Otsu's Method", 1.6e-7], ['Logistic Regression', 0.023e-3], ['LochNet', 6.7e-3]],
                      columns=['Algorithm', 'Classification Time (seconds)'])
    plot = sns.barplot(data=df, x='Algorithm', y='Classification Time (seconds)', log=True)
    plot.bar_label(plot.containers[0])
    plot.set_title('Average Image Classification Time Comparison')
    plt.savefig('time.svg', dpi=2540)  # 2540
    plt.show()


def memory_plot():
    df = pd.DataFrame(data=[["Otsu's Method", 0], ['Logistic Regression', 730], ['LochNet', 396e6]],
                      columns=['Algorithm', 'Memory Usage (bytes)'])
    plot = sns.barplot(data=df, x='Algorithm', y='Memory Usage (bytes)', log=True)
    plot.bar_label(plot.containers[0])
    plot.set_title('Memory Usage Comparison')
    plt.savefig('memory.svg', dpi=2540)  # 2540
    plt.show()


def deep_scatter():
    deep = pd.read_csv('Deep_Results.csv')

    deep.loc[deep['Activation Function'] == 'leaky_relu', 'Activation Function'] = 'Leaky ReLU'
    deep.loc[deep['Activation Function'] == 'relu', 'Activation Function'] = 'ReLU'

    deep = deep[deep['Iteration'] == 10]
    plot = sns.scatterplot(data=deep, x='Model Name', y='F1', size='Hidden Layer Size',
                           hue='Activation Function', sizes=[80, 150, 300])
    plot.axhline(baseline_f1, ls='--', c='k', label='Baseline')
    plot.set(ylim=(0.72, 0.875))
    plot.set_title('Deep Learning: F1-score for different combinations')
    plt.tight_layout()
    plt.savefig('deep_scatter.png', dpi=2540)
    plt.show()


if __name__ == '__main__':
    method_plot()
