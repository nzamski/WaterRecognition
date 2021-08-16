import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.linear_model import SGDClassifier


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
    # returns shrunken squared RGB dataframe
    basic_values = df[['r', 'g', 'b']].values / 255
    square = basic_values**2
    together = np.concatenate([basic_values, square], axis=1)
    return together


def train_model():
    # define model classifier
    classifier = SGDClassifier(loss='epsilon_insensitive', max_iter=5, epsilon=0.2)
    # retrieve train and test files
    train, test = get_train_test()
    squared_train = preprocessing(train)
    squared_test = preprocessing(test)
    accuracy_per_iter = list()
    for _ in tqdm(range(10)):
        # feed the model with the train files
        classifier.fit(squared_train, train['value'].values, sample_weight=train['count'])
        # make prediction using the test files
        prediction = classifier.predict(squared_test)
        # store prediction and accuracy of the prediction
        test['prediction'] = prediction
        test['correct_prediction'] = (test['value'] == test['prediction'])
        # export test data frame to CSV
        test.to_csv('test_per_pixel_SGD.csv', index=False)
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
         'ITERATION': [5*i for i in range(1, len(accuracy_per_iter)+1)]}
    )
    plot = sns.lineplot(data=df, x='ITERATION', y='ACCURACY')
    plot.set(ylim=(0, 1))
    plot.set_title('Stochastic Gradient Descent')
    plot.figure.savefig('SGD Accuracy Per Iteration (Squared).png', dpi=720)
    plt.show()


if __name__ == '__main__':
    train_model()
