import pickle
import numpy as np
import pandas as pd

from tqdm import tqdm
from datetime import datetime
from sklearn.metrics import f1_score
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
    f1_per_iter, train_times, test_times = list(), list(), list()
    # pre-training
    train_start = datetime.now()
    classifier.fit(np.random.rand(2, 3), np.array([0, 1]))
    train_end = datetime.now()
    train_seconds = (train_end - train_start).total_seconds()
    test_start = datetime.now()
    prediction = classifier.predict(normalized_test)
    test_end = datetime.now()
    test_seconds = (test_end - test_start).total_seconds()
    train_times.append(train_seconds)
    test_times.append(test_seconds)
    test['prediction'] = prediction
    test['correct_prediction'] = (test['value'] == test['prediction'])
    f1 = f1_score(test['value'], test['prediction'], sample_weight=test['count'])
    f1_per_iter.append(f1)
    for _ in tqdm(range(int(9 / classifier.max_iter))):
        # feed the model with the train files
        train_start = datetime.now()
        classifier.fit(normalized_train, train['value'].values, sample_weight=train['count'])
        train_end = datetime.now()
        train_seconds = (train_end - train_start).total_seconds()
        # make prediction using the test files
        test_start = datetime.now()
        prediction = classifier.predict(normalized_test)
        test_end = datetime.now()
        test_seconds = (test_end - test_start).total_seconds()
        train_times.append(train_seconds)
        test_times.append(test_seconds)
        # store prediction and accuracy of the prediction
        test['prediction'] = prediction
        test['correct_prediction'] = (test['value'] == test['prediction'])
        # find f1-score
        f1 = f1_score(test['value'], test['prediction'], sample_weight=test['count'])
        f1_per_iter.append(f1)
    df = pd.DataFrame(
        {'Iteration': [classifier.max_iter*i + 1 for i in range(len(f1_per_iter))],
         'F1': f1_per_iter,
         'Train Times': train_times,
         'Test Times': test_times}
    )
    df.to_csv(file, index=False)

    file_name = f'{title}.pkl'
    with open(file_name, "wb") as open_file:
        pickle.dump(classifier, open_file)


if __name__ == '__main__':
    classifiers = [LogisticRegression(max_iter=1, warm_start=True),
                   SGDClassifier(loss='hinge', max_iter=1, warm_start=True, alpha=0.05)]

    figure_outputs = ['Logit F1 Per Iteration.png',
                      'SGD F1 Per Iteration.png']

    file_outputs = ['Logit_Results.csv',
                    'SGD_Results.csv']

    plot_titles = ['Logistic Regression',
                   'Stochastic Gradient Descent']

    for classifier, fig, file, title in zip(classifiers, figure_outputs, file_outputs, plot_titles):
        if title == 'Logistic Regression':
            train_model(classifier, fig, file, title)
