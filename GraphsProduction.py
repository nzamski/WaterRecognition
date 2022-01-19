import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os


def plot_performance(title):
    otsu = pd.read_csv('Otsu_Results.csv')
    logit = pd.read_csv('test_per_pixel_Logit.csv')
    sgd = pd.read_csv('test_per_pixel_SGD.csv')
    deep = pd.read_csv('Water_Bodies_Results.csv')

    # otsu = otsu.groupby('coeff').sum(['correct_pixels', 'image_size']).reset_index()
    # otsu['accuracy'] = otsu['correct_pixels'] / otsu['image_size']
    # otsu['name'] = 'Otsu'
    # otsu.sort_values('accuracy', ascending=False, inplace=True)
    # otsu = otsu.head(1)
    # otsu.columns = ['name', 'Accuracy']
    #
    # logit['name'] = 'Logit'
    # logit.sort_values('ACCURACY', ascending=False, inplace=True)
    # logit = logit.head(1)
    # logit.columns = ['name', 'Accuracy']
    #
    # sgd['name'] = 'SGD'
    # sgd.sort_values('ACCURACY', ascending=False, inplace=True)
    # sgd = sgd.head(1)
    # sgd = sgd[['name', 'ACCURACY']]
    # sgd.columns = ['name', 'Accuracy']
    #
    # deep['name'] = 'Deep'
    # deep.sort_values('Accuracy', ascending=False, inplace=True)
    # deep = deep.head(1)
    # deep = deep[['name', 'Accuracy']]

    # plot = sns.boxplot(data=df, x='Model Name', y='Iteration Training Seconds', hue='Activation Function')
    # fig = plot.get_figure()
    # plot.set(ylim=(0, 1))
    # plot.set_title(title)
    # plt.savefig(title+'.png', dpi=720)
    # plt.show()

    plot = sns.lineplot(data=sgd, x='ITERATION', y='ACCURACY')
    plot.set(ylim=(0, 1))
    plot.set_title(title)
    fig = plot.get_figure()
    fig.savefig(title)
    plt.show()


if __name__ == '__main__':
    os.chdir('D:/Noam/Desktop/WaterRecognition')
    plot_performance('SGD Accuracy per iteration')
