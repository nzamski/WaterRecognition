import pandas as pd

from Models import *
from tqdm import tqdm
from torch import optim
from datetime import datetime
from WaterDataset import get_train_test_loaders
from sklearn.metrics import accuracy_score, f1_score


def fit_model(model, model_parameters, loss_function, optimizer, batch_size, image_normalized_length, num_of_epochs):
    # retrieve train and test files
    train_loader, test_loader = get_train_test_loaders(batch_size, image_normalized_length)
    # if GPU is available, prepare it for heavy calculations
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # assign the model
    model = model(*model_parameters).to(device)
    # same importance for every pixel, same calculation for every pixel
    loss_reduction = 'sum'
    # pos_weight = torch.ones([(image_normalized_length ** 2) * batch_size]).to(device)
    # set a loss function
    # criterion = loss_function(reduction=loss_reduction, pos_weight=pos_weight)
    criterion = loss_function()
    # set an optimizer
    optimizer = optimizer(model.parameters(), lr=0.001)
    # MODEL TRAINING
    model.train()
    for epoch in range(1, num_of_epochs + 1):
        # start counting epoch duration
        epoch_start = datetime.now()
        # initiate epoch loss
        epoch_loss = 0
        # iterate through all data pairs
        for image, mask in tqdm(train_loader):
            # convert input pixel to tensor
            x = image.float().to(device)
            # convert target to tensor
            tag = mask.flatten().long().to(device)
            # reset all gradients
            optimizer.zero_grad()
            # save current prediction
            prediction = model(x)
            # if len(prediction) != len(pos_weight):
            #     pos_weight = torch.ones([len(prediction)]).to(device)
            #     criterion = loss_function(reduction=loss_reduction, pos_weight=pos_weight)
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
            x = x.to(device)
            real.append(y)
            probabilities = model(x)
            batch_predicted = torch.argmax(probabilities, dim=1)
            predicted.append(batch_predicted)
        real = torch.cat(real).flatten()
        predicted = torch.cat(predicted).flatten().detach().cpu()
        # calculate accuracy and f1 score
        accuracy = accuracy_score(real, predicted)
        f1 = f1_score(real, predicted)
        # append results to csv file
        hyperparameters = {'Input Image Length': image_normalized_length,
                           'Hidden Layer Size': hidden_layer_size,
                           'Activation Function': str(model_parameters[2].__name__),
                           'Optimizer': str(type(optimizer)),
                           'Loss Function': str(loss_function)}
        df = pd.DataFrame({'Model Name': ['Hidden1'],
                           'Iteration': [epoch],
                           'Hyperparameters': str(hyperparameters),
                           'Loss': [epoch_loss],
                           'Accuracy': [accuracy],
                           'F1': [f1],
                           'Iteration Training Seconds': [epoch_seconds]})
        df.to_csv('Water_Bodies_Results.csv', index=False, mode='a', header=False)
        print(df)


if __name__ == '__main__':
    # models = (Hidden1, Hidden2, Conv1, Conv2, Conv3)
    models = [Conv1]
    activation_funcs = (f.relu, f.leaky_relu, f.sigmoid)
    hidden_layer_sizes = (50, 100, 200)
    batch_sizes = (2, 8)
    optimizers = (optim.Adam, optim.SGD)

    image_normalized_length = 100
    num_of_epochs = 10
    loss_func = nn.CrossEntropyLoss
    kernel_size = 3

    # train models with varying hyperparameters
    for model in models:
        for activation_func in activation_funcs:
            for hidden_layer_size in hidden_layer_sizes:
                for batch_size in batch_sizes:
                    for optimizer in optimizers:
                        model_parameters = (image_normalized_length, hidden_layer_size, activation_func, kernel_size)
                        fit_model(model, model_parameters, loss_func, optimizer,
                                  batch_size, image_normalized_length, num_of_epochs)
