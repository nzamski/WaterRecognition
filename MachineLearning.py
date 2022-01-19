import numpy as np
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
    # pre-training
    classifier.fit(np.random.rand(2, 3), np.array([0, 1]))
    prediction = classifier.predict(normalized_test)
    test['prediction'] = prediction
    test['correct_prediction'] = (test['value'] == test['prediction'])
    results = test.groupby('correct_prediction')['count'].sum().reset_index(name='count')
    print(results)
    accuracy = results[results['correct_prediction'] == True]['count'].item() / results['count'].sum()
    accuracy_per_iter.append(accuracy)
    print(accuracy)
    for _ in tqdm(range(int(150 / classifier.max_iter))):
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

    # original distribution in dataset
    baseline = 0.676613

    # show a plot of accuracy per iteration
    df = pd.DataFrame(
        {'ACCURACY': accuracy_per_iter,
         'ITERATION': [classifier.max_iter*i for i in range(1, len(accuracy_per_iter)+1)],
         'MODEL TYPE': [title for _ in range(1, len(accuracy_per_iter) + 1)]}
    )
    # export test data frame to CSV
    baseline_df = pd.DataFrame(
        {'ACCURACY': [baseline for _ in range(len(accuracy_per_iter))],
         'ITERATION': [classifier.max_iter * i for i in range(1, len(accuracy_per_iter) + 1)],
         'MODEL TYPE': ['Baseline' for _ in range(len(accuracy_per_iter))]}
    )

    df = pd.concat([df, baseline_df])
    df.to_csv(file, index=False)


if __name__ == '__main__':
    classifiers = [LogisticRegression(max_iter=1, warm_start=True),
                   SGDClassifier(loss='hinge', max_iter=1, warm_start=True, alpha=2e10)]

    figure_outputs = ['Logit Accuracy Per Iteration.png',
                      'SGD Accuracy Per Iteration.png']

    file_outputs = ['test_per_pixel_Logit.csv',
                    'test_per_pixel_SGD.csv']

    plot_titles = ['Logistic Regression',
                   'Stochastic Gradient Descent']

    for classifier, fig, file, title in zip(classifiers, figure_outputs, file_outputs, plot_titles):
        train_model(classifier, fig, file, title)
