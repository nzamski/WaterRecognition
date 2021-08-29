import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.linear_model import LogisticRegression, SGDClassifier


def get_train_test(train_path='rgb_train.csv', test_path='rgb_test.csv'):
    # read data from CSVs
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    # convert mask values to absolute 0 or 1
    train['value'] = (train['value'] < 128).astype('int64')
    test['value'] = (test['value'] < 128).astype('int64')
    # return modified data
    return train, test


def preprocessing(df):
    # returns shrunken RGB dataframe
    basic_values = df[['r', 'g', 'b']].values / 255
    return basic_values


def train_model(classifier, fig, file, title):
    # retrieve train and test files
    train, test = get_train_test()
    normalized_train = preprocessing(train)
    normalized_test = preprocessing(test)
    accuracy_per_iter = list()
    for _ in tqdm(range(50)):
        # feed the model with the train files
        classifier.fit(normalized_train, train['value'].values, sample_weight=train['count'])
        # make prediction using the test files
        prediction = classifier.predict(normalized_test)
        # store prediction and accuracy of the prediction
        test['prediction'] = prediction
        test['correct_prediction'] = (test['value'] == test['prediction'])
        # print aggregation of predictions
        results = test.groupby('correct_prediction')['count'].sum().reset_index(name='count')
        print(results)
        # print accuracy value
        accuracy = results[results['correct_prediction'] == True]['count'].item() / results['count'].sum()
        accuracy_per_iter.append(accuracy)
        print(accuracy)
    # show a plot of accuracy per iteration
    df = pd.DataFrame(
        {'ACCURACY': accuracy_per_iter,
         'ITERATION': [i for i in range(1, len(accuracy_per_iter)+1)]}
    )
    # export test data frame to CSV
    df.to_csv(file, index=False)

    plot = sns.lineplot(data=df, x='ITERATION', y='ACCURACY')
    plot.set(ylim=(0, 1))
    plot.set_title(title)
    plot.figure.savefig(fig, dpi=720)
    plt.show()


if __name__ == '__main__':
    classifiers = [LogisticRegression(max_iter=1, warm_start=True),
                   SGDClassifier(loss='epsilon_insensitive', max_iter=1, epsilon=0.2)]

    figure_outputs = ['Logit Accuracy Per Iteration.png',
                      'SGD Accuracy Per Iteration.png']

    file_outputs = ['test_per_pixel_Logit.csv',
                    'test_per_pixel_SGD.csv']

    plot_titles = ['Logistic Regression',
                   'Stochastic Gradient Descent']

    for classifier, fig, file, title in zip(classifiers, figure_outputs, file_outputs, plot_titles):
        train_model(classifier, fig, file, title)
