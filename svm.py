import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
import pandas as pd


def get_train_test(train_path='rgb_train.csv', test_path='rgb_test.csv'):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    train['value'] = (train['value'] > 128).astype('int64')
    test['value'] = (test['value'] > 128).astype('int64')
    return train, test


def train_model():
    classifier = SVC()
    train, test = get_train_test()
    train = train
    test = test
    classifier.fit(train[['r', 'g', 'b']].values, train['value'].values, sample_weight=train['count'])
    prediction = classifier.predict(test[['r', 'g', 'b']].values)
    test['prediction'] = prediction
    test['correct_prediction'] = (test['value'] == test['prediction'])
    test.to_csv('test_per_pixel_SVM.csv', index=False)
    test = test.groupby('correct_prediction')['count'].sum().reset_index(name='count')
    print(test)
    print(test[test['correct_prediction'] == True]['count'].item() / test['count'].sum())


if __name__ == '__main__':
    train_model()
