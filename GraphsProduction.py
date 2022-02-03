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

    otsu = pd.DataFrame({'Iteration': [i + 1 for i in range(0, 10)],
                         'F1': [otsu['F1'].median() for _ in range(0, 10)]})

    otsu['Model Name'] = "Otsu's Method (median)"
    logit['Model Name'] = 'Logistic Regression'
    deep['Model Name'] = 'LochNet (ReLU)'
    df = pd.concat([otsu, logit, deep]).set_index([[i for i in range(0, 30)]])

    plot = sns.lineplot(data=df, x='Iteration', y='F1', hue='Model Name')
    plot.axhline(baseline_f1, ls='--', c='k', label='Baseline')
    plot.set(ylim=(0, 1))
    plot.set_title('All 3 methods: F1-score for best models')
    plt.savefig('comparison.png', dpi=720)  # or 2540
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

    plot = sns.lineplot(data=loch_net, x='Iteration', y='F1', hue='Activation Function')
    plot.axhline(baseline_f1, ls='--', c='r', label='Baseline')
    plot.set(ylim=(0, 1))
    plot.set_title('Deep Learning: F1-score for different activation functions')
    plt.savefig('deep.png', dpi=720)
    plt.show()


if __name__ == '__main__':
    comparison()
