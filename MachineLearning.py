import numpy as np
import pandas as pd

from tqdm import tqdm
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
    f1_per_iter = list()
    # pre-training
    classifier.fit(np.random.rand(2, 3), np.array([0, 1]))
    prediction = classifier.predict(normalized_test)
    test['prediction'] = prediction
    test['correct_prediction'] = (test['value'] == test['prediction'])
    f1 = f1_score(test['value'], test['prediction'], sample_weight=test['count'])
    f1_per_iter.append(f1)
    for _ in tqdm(range(int(9 / classifier.max_iter))):
        # feed the model with the train files
        classifier.fit(normalized_train, train['value'].values, sample_weight=train['count'])
        # make prediction using the test files
        prediction = classifier.predict(normalized_test)
        # store prediction and accuracy of the prediction
        test['prediction'] = prediction
        test['correct_prediction'] = (test['value'] == test['prediction'])
        # find f1-score
        f1 = f1_score(test['value'], test['prediction'], sample_weight=test['count'])
        f1_per_iter.append(f1)
    df = pd.DataFrame(
        {'Iteration': [classifier.max_iter*i + 1 for i in range(len(f1_per_iter))], 'F1': f1_per_iter})
    df.to_csv(file, index=False)


if __name__ == '__main__':
    classifiers = [LogisticRegression(max_iter=1, warm_start=True),
                   SGDClassifier(loss='hinge', max_iter=1, warm_start=True, alpha=0.1)]

    figure_outputs = ['Logit F1 Per Iteration.png',
                      'SGD F1 Per Iteration.png']

    file_outputs = ['test_per_pixel_Logit.csv',
                    'test_per_pixel_SGD.csv']

    plot_titles = ['Logistic Regression',
                   'Stochastic Gradient Descent']

    for classifier, fig, file, title in zip(classifiers, figure_outputs, file_outputs, plot_titles):
        if title == 'Stochastic Gradient Descent':
            train_model(classifier, fig, file, title)
