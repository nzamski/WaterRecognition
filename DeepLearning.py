import os
import cv2
import pandas as pd

from DeepModels import *
from tqdm import tqdm
from torch import optim
from datetime import datetime
from torchvision.utils import save_image
from torchvision.transforms import Resize
from DataLoader import get_train_test_loaders
from sklearn.metrics import accuracy_score, f1_score


def get_img_index(path):
    index = str(path).split('_')[-1].split('.')[0]
    return index


def fit_model(model, model_parameters, loss_function, optimizer, batch_size, image_normalized_length, num_of_epochs):
    # retrieve train and test files
    train_loader, test_loader = get_train_test_loaders(batch_size, image_normalized_length)
    # if GPU is available, prepare it for heavy calculations
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # assign the model
    model = model(*model_parameters).to(device)
    # assign the loss function
    criterion = loss_function()
    # set an optimizer
    optimizer = optimizer(model.parameters(), lr=3e-4)
    for epoch in range(1, num_of_epochs + 1):
        # MODEL TRAINING
        model.train()
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
        real = torch.cat(real).reshape(-1)
        predicted = torch.cat(predicted).reshape(-1).detach().cpu()
        # calculate accuracy and f1 score
        accuracy = accuracy_score(real, predicted)
        f1 = f1_score(real, predicted)
        # append results to csv file
        df = pd.DataFrame({'Model Name': [model.__class__.__name__],
                           'Iteration': [epoch],
                           'Input Image Length': [image_normalized_length],
                           'Hidden Layer Size': [hidden_layer_size],
                           'Batch Size': [batch_size],
                           'Activation Function': [str(model_parameters[2].__name__)],
                           'Optimizer': [str(type(optimizer))],
                           'Loss Function': [str(loss_function)],
                           'Loss': [epoch_loss],
                           'Accuracy': [accuracy],
                           'F1': [f1],
                           'Iteration Training Seconds': [epoch_seconds]})
        df.to_csv('drive/MyDrive/Water_Bodies_Results.csv', index=False, mode='a', header=False)
        print(df)
    # save a prediction
    # with torch.no_grad():
    #     for i, image, mask in enumerate(test_loader):
    #         name = ''
    #         x = image.float().to(device)
    #         tag = mask.flatten().long().to(device)
    #         prediction = model(x)
    #         file_name, _ = test_loader.dataset.samples[i]
    #         file_index = get_img_index(file_name)
    #         save_prediction(prediction, file_index, name)
    torch.cuda.empty_cache()


def save_prediction(prediction, index, name):
    # convert two-valued pixels to single-max-value
    prediction = torch.argmax(prediction, dim=1)
    # reshape the flattened prediction to a matrix
    prediction = prediction.reshape(100, 100)
    # get original size
    source_path = f'{os.getcwd()}{os.sep}Water Bodies Dataset{os.sep}Images{os.sep}water_body_{index}.jpg'
    width, height = cv2.imread(source_path).shape
    # resize prediction to original source size
    prediction = Resize(prediction, size=(width, height))
    prediction_path = f'{os.getcwd()}{os.sep}Deep Images{os.sep}{name}_{index}.jpg'
    # save the prediction
    save_image(prediction, prediction_path)


if __name__ == '__main__':
    models = (Hidden1, Hidden2, Conv1, Conv2, Conv3)
    activation_funcs = (f.relu, f.leaky_relu, f.sigmoid)
    hidden_layer_sizes = (2_500, 10_000, 40_000)
    batch_sizes = (4, 16)
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
                        model_parameters = (image_normalized_length, hidden_layer_size, activation_func)
                        fit_model(model, model_parameters, loss_func, optimizer,
                                  batch_size, image_normalized_length, num_of_epochs)
