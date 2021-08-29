import pandas as pd

from Models import *
from tqdm import tqdm
from torch import optim
from datetime import datetime
from WaterDataset import get_train_test_loaders
from sklearn.metrics import accuracy_score, f1_score


def fit_model(model, model_parameters, loss_function, optimizer, batch_size, image_normalized_length):
    # retrieve train and test files
    train_loader, test_loader = get_train_test_loaders(batch_size, image_normalized_length)
    # assign the model
    model = model(*model_parameters)

    # same importance for every pixel, same calculation for every pixel
    loss_reduction = 'sum'
    pos_weight = torch.ones([(image_normalized_length ** 2) * batch_size])

    # set a loss function
    criterion = loss_function(reduction=loss_reduction, pos_weight=pos_weight)
    # set an optimizer
    optimizer = optimizer(model.parameters(), lr=0.001)
    # MODEL TRAINING
    model.train()
    for epoch in range(10):
        # start counting epoch duration
        epoch_start = datetime.now()
        # initiate epoch loss
        epoch_loss = 0
        # iterate through all data pairs
        for image, mask in tqdm(train_loader):
            # convert input pixel to tensor
            x = torch.tensor(image).float()
            # convert target to tensor
            tag = torch.tensor(mask, dtype=torch.float).flatten()
            # reset all gradients
            optimizer.zero_grad()
            # save current prediction
            prediction = model(x).view(-1)
            if len(prediction) != len(pos_weight):
                pos_weight = torch.ones([len(prediction)])
                criterion = loss_function(reduction=loss_reduction, pos_weight=pos_weight)
            # activate loss function, calculate loss
            loss = criterion(prediction, tag)
            # back propagation
            loss.backward()
            optimizer.step()
            # update epoch loss
            epoch_loss += loss.item()
        # stop counting epoch duration
        epoch_end = datetime.now()
        epoch_seconds = (epoch_end - epoch_start).total_seconds()
        # MODEL EVALUATION
        model.eval()
        # collect predicted results and real results
        predicted, real = list(), list()
        for x, y in tqdm(test_loader):
            real.append(y)
            probabilities = model(x)
            batch_predicted = torch.argmax(probabilities, dim=1)
            predicted.append(batch_predicted)
        real = torch.cat(real).flatten()
        predicted = torch.cat(predicted).flatten()
        # calculate accuracy and f1 score
        accuracy = accuracy_score(real, predicted)
        f1 = f1_score(real, predicted)
        # append results to csv file
        df = pd.DataFrame({'Model Name': ['Hidden1'],
                           'Iteration': [epoch],
                           'Hyperparameters': [f'''"input_image_length": 5
                                 "hidden_layer_size": 10
                                 "activation": "f.relu"
                                 "optimizer": "optim.Adam"
                                 "loss_function": "nn.CrossEntropyLoss"'''],
                           'Loss': [epoch_loss],
                           'Accuracy': [accuracy],
                           'F1': [f1],
                           'Iteration Training Seconds': [epoch_seconds]})
        df.to_csv('Water_Bodies_Results.csv', index=False, mode='a', header=False)
        print(df)


if __name__ == '__main__':
    loss_func = nn.BCEWithLogitsLoss
    optimizers = (optim.Adam, optim.SGD)
    activation_funcs = (f.relu, f.leaky_relu, f.sigmoid)
    image_normalized_length = 100
    batch_size = 2
    hidden_layer_size = 50

    # train the model
    model_parameters = (image_normalized_length, hidden_layer_size, activation_funcs[0])
    fit_model(Hidden1, model_parameters, loss_func, optimizers[0], batch_size, image_normalized_length)
