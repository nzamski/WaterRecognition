from sklearn.linear_model import LogisticRegression
import pandas as pd


def get_train_test(train_path='rgb_train.csv', test_path='rgb_test.csv'):
    # read data from CSVs
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    # convert mask values to absolute 0 or 1
    train['value'] = (train['value'] > 128).astype('int64')
    test['value'] = (test['value'] > 128).astype('int64')
    # return modified data
    return train, test


def train_model():
    # define model classifier
    classifier = LogisticRegression()
    # retrieve train and test files
    train, test = get_train_test()
    # feed the model with the train files
    classifier.fit(train[['r', 'g', 'b']].values, train['value'].values, sample_weight=train['count'])
    # make prediction using the test files
    prediction = classifier.predict(test[['r', 'g', 'b']].values)
    # store prediction and accuracy of the prediction
    test['prediction'] = prediction
    test['correct_prediction'] = (test['value'] == test['prediction'])
    # export test data frame to CSV
    test.to_csv('test_per_pixel_Logistic.csv', index=False)
    # print test data frame and accuracy
    test = test.groupby('correct_prediction')['count'].sum().reset_index(name='count')
    print(test)
    print(test[test['correct_prediction'] == True]['count'].item() / test['count'].sum())


if __name__ == '__main__':
    train_model()
